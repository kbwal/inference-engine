import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bn in [32, 64, 128]
        for w in [4, 8]
        for st in [2, 3]
    ],
    key=["num_batches", "max_len"],
)
@triton.jit
def decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    key_mask_ptr,
    cache_pos_ptr,
    partial_accumulation_ptr,
    partial_max_ptr,
    partial_running_sum_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_mb,
    scale,
    num_batches,
    max_len,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    d_offsets = tl.arange(0, D)
    h_offsets = tl.arange(0, 16)
    h_valid = h_offsets < GQA_RATIO

    # output is (B, H, 1, D), so index by (B, H)
    b_idx = tl.program_id(axis=0)  # batch
    h_idx = tl.program_id(axis=1)  # head

    q = tl.load(
        q_ptr
        + b_idx * stride_qb
        + (h_idx * GQA_RATIO + h_offsets)[:, None] * stride_qh
        + d_offsets[None, :],
        mask=h_valid[:, None],
        other=0.0,
    )  # (16, D), last 14 are 0s. this makes it use more compute, but this kernel is memory bound

    cache_pos = tl.load(cache_pos_ptr)  # [1]
    n_end = cache_pos + 1

    split_idx = tl.program_id(axis=2)
    split_len = tl.cdiv(tl.cdiv(n_end, NUM_SPLITS), BLOCK_N) * BLOCK_N
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, n_end)

    kv_bh_offsets = b_idx * stride_kb + h_idx * stride_kh

    max_old = tl.full((16,), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((16,), dtype=tl.float32)
    accumulator = tl.zeros((16, D), dtype=tl.float32)

    for start_n in range(split_start, split_end, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        in_cache = n_offsets < split_end  # [BLOCK_N]

        kv_nd_offsets = n_offsets[:, None] * stride_kn + d_offsets[None, :]
        k = tl.load(
            k_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        )  # (BLOCK_N, D)
        v = tl.load(
            v_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        )  # (BLOCK_N, D)

        key_mask_offsets = b_idx * stride_mb + n_offsets
        key_mask = tl.load(key_mask_ptr + key_mask_offsets, mask=in_cache, other=False)[
            None, :
        ]  # [1, BLOCK_N]

        mask = in_cache[None, :] & key_mask  # [1, BLOCK_N]

        attn_scores = tl.dot(q, tl.trans(k)) * scale  # [16, BLOCK_N]
        attn_scores = tl.where(mask, attn_scores, float("-inf"))

        max_score = tl.max(attn_scores, axis=1)
        max_new = tl.maximum(max_score, max_old)
        safe_max_new = tl.where(max_new == float("-inf"), 0.0, max_new)

        alpha = tl.exp(max_old - safe_max_new)
        numerator = tl.exp(attn_scores - safe_max_new[:, None])
        running_sum = running_sum * alpha + tl.sum(numerator, axis=1)

        weighted_sum = tl.dot(numerator.to(v.dtype), v)  # (16, D)
        accumulator = accumulator * alpha[:, None] + weighted_sum

        max_old = max_new

    q_heads = h_idx * GQA_RATIO + h_offsets

    if NUM_SPLITS == 1:
        out = accumulator / running_sum[:, None]
        tl.store(
            output_ptr
            + b_idx * stride_ob
            + (h_idx * GQA_RATIO + h_offsets)[:, None] * stride_oh
            + d_offsets[None, :],
            out,
            mask=h_valid[:, None],
        )
    else:
        # we only saw a small slice, so we can't store the entire softmax in out
        HQ = tl.num_programs(axis=1) * GQA_RATIO
        rows = (b_idx * HQ + q_heads) * NUM_SPLITS + split_idx  # (16)
        tl.store(
            partial_accumulation_ptr + rows[:, None] * D + d_offsets[None, :],
            accumulator,
            mask=h_valid[:, None],
        )
        tl.store(partial_max_ptr + rows, max_old, mask=h_valid)
        tl.store(partial_running_sum_ptr + rows, running_sum, mask=h_valid)


@triton.jit
def decode_attention_combine_kernel(
    partial_accumulator_ptr,
    partial_max_ptr,
    partial_running_sum_ptr,
    output_ptr,
    stride_ob,
    stride_oh,
    HQ: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b_idx = tl.program_id(axis=0)
    q_head = tl.program_id(axis=1)
    s = tl.arange(0, NUM_SPLITS)
    d_offsets = tl.arange(0, D)
    rows = (b_idx * HQ + q_head) * NUM_SPLITS + s

    maxs = tl.load(partial_max_ptr + rows)  # (S)
    running_sum = tl.load(partial_running_sum_ptr + rows)  # (S)
    accumulator = tl.load(
        partial_accumulator_ptr + rows[:, None] * D + d_offsets[None, :]
    )  # (S, D)

    max_score = tl.max(maxs, axis=0)
    numerator = tl.where(maxs == float("-inf"), 0.0, tl.exp(maxs - max_score))  # (S)
    out = tl.sum(accumulator * numerator[:, None], axis=0) / tl.sum(
        running_sum * numerator, axis=0
    )  # type: ignore
    tl.store(output_ptr + b_idx * stride_ob + q_head * stride_oh + d_offsets, out)


def decode_attention(
    q: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    key_mask: torch.Tensor,
    cache_pos: torch.Tensor,
    scale,
):
    k_cache, v_cache = kv_cache
    B, Hq, T, D = q.shape
    Hkv = k_cache.shape[1]
    assert T == 1, "somehow T != 1 in the custom decode attention kernel!"
    out = torch.empty(B, Hq, 1, D, device=q.device, dtype=q.dtype)

    # we have 82 SMs, so if B is too small then not all are used
    # use num_splits to split the work when B is low
    num_splits = min(16, triton.next_power_of_2(max(1, 128 // (B * Hkv))))
    if num_splits > 1:
        partial_accumulator = torch.empty(
            B, Hq, num_splits, D, device=q.device, dtype=torch.float32
        )
        partial_maxes = torch.empty(
            B, Hq, num_splits, device=q.device, dtype=torch.float32
        )
        partial_running_sum = torch.empty(
            B, Hq, num_splits, device=q.device, dtype=torch.float32
        )
    else:
        partial_accumulator = partial_maxes = partial_running_sum = (
            out  # unused, triton just needs a tensor
        )

    decode_attention_kernel[(B, Hkv, num_splits)](
        q,
        k_cache,
        v_cache,
        out,
        key_mask,
        cache_pos,
        partial_accumulator,
        partial_maxes,
        partial_running_sum,
        q.stride(dim=0),
        q.stride(dim=1),
        k_cache.stride(dim=0),
        k_cache.stride(dim=1),
        k_cache.stride(dim=2),
        out.stride(dim=0),
        out.stride(dim=1),
        key_mask.stride(dim=0),
        scale,
        num_batches=B,
        max_len=triton.next_power_of_2(k_cache.shape[2]),
        GQA_RATIO=Hq // Hkv,
        D=D,
        NUM_SPLITS=num_splits,
    )
    if num_splits > 1:
        decode_attention_combine_kernel[(B, Hq)](
            partial_accumulator,
            partial_maxes,
            partial_running_sum,
            out,
            out.stride(0),
            out.stride(1),
            HQ=Hq,  # type: ignore
            D=D,  # type: ignore
            NUM_SPLITS=num_splits,  # type: ignore
        )
    return out


@triton.jit
def fused_rope_kv_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_norm_ptr,
    k_norm_ptr,
    output_ptr,
    cache_pos_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cos_ptr,
    sin_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_cosb,
    stride_sinb,
    stride_cb,
    stride_ch,
    stride_cn,
    stride_ob,
    stride_oh,
    eps,
    HQ: tl.constexpr,
    D: tl.constexpr,
):
    b_idx = tl.program_id(axis=0)
    h_idx = tl.program_id(axis=1)

    half_d_offsets = tl.arange(0, D // 2)  # load half
    if h_idx < HQ:
        q_offset = b_idx * stride_qb + h_idx * stride_qh

        x1 = tl.load(q_ptr + q_offset + half_d_offsets).to(tl.float32)  # [D // 2]
        x2 = tl.load(q_ptr + q_offset + half_d_offsets + D // 2).to(tl.float32)

        rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        x1 = x1 * rstd * tl.load(q_norm_ptr + half_d_offsets).to(tl.float32)
        x2 = x2 * rstd * tl.load(q_norm_ptr + D // 2 + half_d_offsets).to(tl.float32)
    else:
        k_offset = b_idx * stride_kb + (h_idx - HQ) * stride_kh

        x1 = tl.load(k_ptr + k_offset + half_d_offsets).to(tl.float32)  # [D // 2]
        x2 = tl.load(k_ptr + k_offset + half_d_offsets + D // 2).to(tl.float32)

        rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        x1 = x1 * rstd * tl.load(k_norm_ptr + half_d_offsets).to(tl.float32)
        x2 = x2 * rstd * tl.load(k_norm_ptr + D // 2 + half_d_offsets).to(tl.float32)

    cos = tl.load(cos_ptr + b_idx * stride_cosb + half_d_offsets).to(
        tl.float32
    )  # [D // 2]
    sin = tl.load(sin_ptr + b_idx * stride_sinb + half_d_offsets).to(tl.float32)

    out1 = x1 * cos - x2 * sin  # [D // 2]
    out2 = x1 * sin + x2 * cos
    out = tl.cat(out1, out2)

    if h_idx < HQ:
        tl.store(
            (output_ptr + stride_ob * b_idx + stride_oh * h_idx) + tl.arange(0, D), out
        )
    else:
        j = h_idx - HQ
        pos = tl.load(cache_pos_ptr)
        d_offsets = tl.arange(0, D)
        slot = (
            b_idx * stride_cb + j * stride_ch + pos * stride_cn
        )  # same strides for k_cache and v_cache

        tl.store(k_cache_ptr + slot + d_offsets, out)
        v = tl.load(v_ptr + b_idx * stride_vb + j * stride_vh + d_offsets)
        tl.store(v_cache_ptr + slot + d_offsets, v)


def fused_rope_kv_decode(
    q: torch.Tensor,  # (B, Hq, 1, D)
    k: torch.Tensor,  # (B, Hkv, 1, D)
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,  # (D,)
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,  # (B, 1, 1, D // 2)
    sin: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    cache_pos: torch.Tensor,
    eps: float,
):
    # returns normed + roped q and writes kv cache
    k_cache, v_cache = kv_cache
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    assert T == 1, "somehow T != 1 in the custom fused rope kernel!"
    assert (
        k_cache.stride() == v_cache.stride()
    ), "kernel uses one set of strides for both caches"
    out = torch.empty(B, Hq, 1, D, device=q.device, dtype=q.dtype)
    fused_rope_kv_decode_kernel[(B, Hq + Hkv)](
        q,
        k,
        v,
        q_norm_weight,
        k_norm_weight,
        out,
        cache_pos,
        kv_cache[0],
        kv_cache[1],
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(0),
        sin.stride(0),
        kv_cache[0].stride(0),
        kv_cache[0].stride(1),
        kv_cache[0].stride(2),
        out.stride(0),
        out.stride(1),
        eps,
        HQ=Hq,  # type: ignore
        D=D,  # type: ignore
    )
    return out
