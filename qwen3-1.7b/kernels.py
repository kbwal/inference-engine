import triton
import triton.language as tl
import torch
from triton.language.extra import libdevice
from dataclasses import dataclass


@dataclass
class LayerKVCache:
    # leading dim is the slot pool, not the batch: sequence b lives in row slot_ids[b]
    k: torch.Tensor  # int8  [pool, Hkv, max_len, D]
    v: torch.Tensor  # int8  [pool, Hkv, max_len, D]
    k_scale: torch.Tensor  # fp32  [pool, Hkv, max_len]
    v_scale: torch.Tensor  # fp32  [pool, Hkv, max_len]
    k_calibration: torch.Tensor  # fp32  [pool, Hkv, 1, D]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bn in [32, 64, 128]
        for w in [4, 8]
        for st in [2, 3]
    ],
    key=["num_batches", "max_len"],
    cache_results=True,
)
@triton.jit
def decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    k_scale_ptr,
    v_scale_ptr,
    cached_lens_ptr,
    partial_accumulation_ptr,
    partial_max_ptr,
    partial_running_sum_ptr,
    slot_ids_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_sb,
    stride_sh,
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

    # output is (B, H, D), so index by (B, H)
    b_idx = tl.program_id(axis=0)  # batch
    h_idx = tl.program_id(axis=1)  # head
    slot = tl.load(slot_ids_ptr + b_idx)

    q = tl.load(
        q_ptr
        + b_idx * stride_qb
        + (h_idx * GQA_RATIO + h_offsets)[:, None] * stride_qh
        + d_offsets[None, :],
        mask=h_valid[:, None],
        other=0.0,
    )  # (16, D), last 14 are 0s. this makes it use more compute, but this kernel is memory bound

    cached_len = tl.load(cached_lens_ptr + b_idx)  # [1]
    n_end = cached_len + 1

    split_idx = tl.program_id(axis=2)
    split_len = tl.cdiv(tl.cdiv(n_end, NUM_SPLITS), BLOCK_N) * BLOCK_N
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, n_end)

    kv_bh_offsets = slot * stride_kb + h_idx * stride_kh

    max_old = tl.full((16,), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((16,), dtype=tl.float32)
    accumulator = tl.zeros((16, D), dtype=tl.float32)

    for start_n in range(split_start, split_end, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        in_cache = n_offsets < split_end  # [BLOCK_N]

        kv_nd_offsets = n_offsets[:, None] * stride_kn + d_offsets[None, :]
        k = tl.load(
            k_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        ).to(
            tl.bfloat16
        )  # (BLOCK_N, D)
        v = tl.load(
            v_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        ).to(
            tl.bfloat16
        )  # (BLOCK_N, D)

        scale_offsets = slot * stride_sb + h_idx * stride_sh + n_offsets
        k_s = tl.load(
            k_scale_ptr + scale_offsets, mask=in_cache, other=0.0
        )  # [BLOCK_N]
        v_s = tl.load(v_scale_ptr + scale_offsets, mask=in_cache, other=0.0)

        attn_scores = tl.dot(q, tl.trans(k)) * (k_s * scale)[None, :]  # [16, BLOCK_N]
        attn_scores = tl.where(in_cache[None, :], attn_scores, float("-inf"))

        max_score = tl.max(attn_scores, axis=1)
        max_new = tl.maximum(max_score, max_old)
        safe_max_new = tl.where(max_new == float("-inf"), 0.0, max_new)

        alpha = tl.exp(max_old - safe_max_new)
        numerator = tl.exp(attn_scores - safe_max_new[:, None])
        running_sum = running_sum * alpha + tl.sum(numerator, axis=1)

        weighted_sum = tl.dot((numerator * v_s[None, :]).to(tl.bfloat16), v)  # (16, D)
        accumulator = accumulator * alpha[:, None] + weighted_sum

        max_old = max_new

    q_heads = h_idx * GQA_RATIO + h_offsets

    if NUM_SPLITS == 1:
        out = accumulator / running_sum[:, None]
        tl.store(
            output_ptr
            + b_idx * stride_ob
            + q_heads[:, None] * stride_oh
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
    q: torch.Tensor,  # [B, Hq, D], one new token per sequence
    kv_cache: LayerKVCache,
    cached_lens: torch.Tensor,  # [B], tokens already cached per sequence
    slot_ids: torch.Tensor,  # [B], kv cache row of each sequence
    scale,
):  # -> [B, Hq, D]
    k_cache, v_cache = kv_cache.k, kv_cache.v
    assert k_cache.stride() == v_cache.stride(), "kernel uses one set of strides for k and v"
    assert kv_cache.k_scale.stride() == kv_cache.v_scale.stride()
    B, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)

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
        kv_cache.k_scale,
        kv_cache.v_scale,
        cached_lens,
        partial_accumulator,
        partial_maxes,
        partial_running_sum,
        slot_ids,
        q.stride(dim=0),
        q.stride(dim=1),
        k_cache.stride(dim=0),
        k_cache.stride(dim=1),
        k_cache.stride(dim=2),
        out.stride(dim=0),
        out.stride(dim=1),
        kv_cache.k_scale.stride(dim=0),
        kv_cache.k_scale.stride(dim=1),
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
            out.stride(dim=0),
            out.stride(dim=1),
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
    cached_lens_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_calibration_ptr,
    cos_ptr,
    sin_ptr,
    slot_ids_ptr,
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
    stride_sb,
    stride_sh,
    stride_calb,
    stride_calh,
    eps,
    HQ: tl.constexpr,
    GQA_RATIO: tl.constexpr,
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

    slot = tl.load(slot_ids_ptr + b_idx)

    if h_idx < HQ:
        # fold channel scale into q: (q * c) * (k / c) == q * k
        calibration_offset = slot * stride_calb + (h_idx // GQA_RATIO) * stride_calh
        c1 = tl.load(k_calibration_ptr + calibration_offset + half_d_offsets)
        c2 = tl.load(k_calibration_ptr + calibration_offset + D // 2 + half_d_offsets)
        o = output_ptr + stride_ob * b_idx + stride_oh * h_idx
        tl.store(o + half_d_offsets, out1 * c1)
        tl.store(o + D // 2 + half_d_offsets, out2 * c2)
    else:
        j = h_idx - HQ
        cached_len = tl.load(cached_lens_ptr + b_idx)
        d_offsets = tl.arange(0, D)
        scale_slot = slot * stride_sb + j * stride_sh + cached_len

        calibration_offset = slot * stride_calb + j * stride_calh
        cache_offset = slot * stride_cb + j * stride_ch + cached_len * stride_cn
        k1 = out1 / tl.load(k_calibration_ptr + calibration_offset + half_d_offsets)
        k2 = out2 / tl.load(
            k_calibration_ptr + calibration_offset + D // 2 + half_d_offsets
        )
        k_amax = tl.maximum(tl.max(tl.abs(k1), axis=0), tl.max(tl.abs(k2), axis=0))
        k_s = tl.maximum(k_amax, 1e-6) / 127
        tl.store(
            k_cache_ptr + cache_offset + half_d_offsets,
            tl.clamp(libdevice.rint(k1 / k_s), -127.0, 127.0).to(tl.int8),
        )
        tl.store(
            k_cache_ptr + cache_offset + D // 2 + half_d_offsets,
            tl.clamp(libdevice.rint(k2 / k_s), -127.0, 127.0).to(tl.int8),
        )
        tl.store(k_scale_ptr + scale_slot, k_s)

        v = tl.load(v_ptr + b_idx * stride_vb + j * stride_vh + d_offsets).to(
            tl.float32
        )
        v_s = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-6) / 127
        tl.store(
            v_cache_ptr + cache_offset + d_offsets,
            tl.clamp(libdevice.rint(v / v_s), -127.0, 127.0).to(tl.int8),
        )
        tl.store(v_scale_ptr + scale_slot, v_s)


def fused_rope_kv_decode(
    q: torch.Tensor,  # (B, Hq, D), one new token per sequence
    k: torch.Tensor,  # (B, Hkv, D)
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,  # (D,)
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,  # (B, 1, D // 2)
    sin: torch.Tensor,
    slot_ids: torch.Tensor,  # [B], list of ints that points to the row holding this sequence's cache
    kv_cache: LayerKVCache,
    cached_lens: torch.Tensor,  # [B], tokens already in the kv cache per sequence
    eps: float,
):
    # returns normed + roped q and writes kv cache
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    assert (
        kv_cache.k.stride() == kv_cache.v.stride()
    ), "kernel uses one set of strides for both caches"
    assert kv_cache.k_scale.stride() == kv_cache.v_scale.stride()
    out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)
    fused_rope_kv_decode_kernel[(B, Hq + Hkv)](
        q,
        k,
        v,
        q_norm_weight,
        k_norm_weight,
        out,
        cached_lens,
        kv_cache.k,
        kv_cache.v,
        kv_cache.k_scale,
        kv_cache.v_scale,
        kv_cache.k_calibration,
        cos,
        sin,
        slot_ids,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(0),
        sin.stride(0),
        kv_cache.k.stride(0),
        kv_cache.k.stride(1),
        kv_cache.k.stride(2),
        out.stride(0),
        out.stride(1),
        kv_cache.k_scale.stride(dim=0),
        kv_cache.k_scale.stride(dim=1),
        kv_cache.k_calibration.stride(dim=0),
        kv_cache.k_calibration.stride(dim=1),
        eps,
        HQ=Hq,  # type: ignore
        GQA_RATIO=Hq // Hkv,  # type: ignore
        D=D,  # type: ignore
    )
    return out


@triton.jit
def quantize_kv_prefill_kernel(
    k_ptr,  # bf16 [N, Hkv, D], all sequences packed along N
    v_ptr,  # bf16 [N, Hkv, D]
    k_cache_ptr,  # int8 [pool, Hkv, max_len, D], sequence b writes row slot_ids[b] from column cached_lens[b]
    v_cache_ptr,  # int8 [pool, Hkv, max_len, D]
    k_scale_ptr,  # fp32 [pool, Hkv, max_len]
    v_scale_ptr,  # fp32 [pool, Hkv, max_len]
    k_calibration_ptr,  # fp32 [pool, Hkv, 1, D], computed on a sequence's first chunk only
    cu_seqlens_ptr,  # int32 [B + 1], sequence i is packed tokens [cu_seqlens[i], cu_seqlens[i+1])
    slot_ids_ptr,
    cached_lens_ptr,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    stride_cb,
    stride_ch,
    stride_cn,
    stride_sb,
    stride_sh,
    stride_calb,
    stride_calh,
    stride_cuseqb,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b_idx = tl.program_id(axis=0)
    h_idx = tl.program_id(axis=1)
    slot = tl.load(slot_ids_ptr + b_idx)
    d_offsets = tl.arange(0, D)

    seq_start = tl.load(cu_seqlens_ptr + b_idx * stride_cuseqb)
    # how many tokens am i writing?
    chunk_len = tl.load(cu_seqlens_ptr + (b_idx + 1) * stride_cuseqb) - seq_start
    # how many tokens do i already have?
    cached_len = tl.load(cached_lens_ptr + b_idx)

    k_base = k_ptr + stride_kt * seq_start + h_idx * stride_kh
    v_base = v_ptr + stride_vt * seq_start + h_idx * stride_vh

    if cached_len == 0:
        calibration = tl.full((D,), 1e-3, dtype=tl.float32)
        for t0 in range(0, chunk_len, BLOCK_T):
            t_offsets = t0 + tl.arange(0, BLOCK_T)
            in_t = t_offsets < chunk_len
            k = tl.load(
                k_base + t_offsets[:, None] * stride_kt + d_offsets[None, :],
                mask=in_t[:, None],
                other=0.0,
            ).to(
                tl.float32
            )  # (BLOCK_T, D)
            calibration = tl.maximum(calibration, tl.max(tl.abs(k), axis=0))
        tl.store(
            k_calibration_ptr + slot * stride_calb + h_idx * stride_calh + d_offsets,
            calibration,
        )
    else:
        calibration = tl.load(
            k_calibration_ptr + slot * stride_calb + h_idx * stride_calh + d_offsets
        )

    for t0 in range(0, chunk_len, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        in_t = t_offsets < chunk_len

        k = (
            tl.load(
                k_base + t_offsets[:, None] * stride_kt + d_offsets[None, :],
                mask=in_t[:, None],
                other=0.0,
            ).to(tl.float32)
            / calibration[None, :]
        )
        k_scale = tl.maximum(tl.max(tl.abs(k), axis=1), 1e-6) / 127  # [BLOCK_T]
        k_quantized = tl.clamp(libdevice.rint(k / k_scale[:, None]), -127.0, 127.0).to(
            tl.int8
        )

        v = tl.load(
            v_base + t_offsets[:, None] * stride_vt + d_offsets[None, :],
            mask=in_t[:, None],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-6) / 127
        v_quantized = tl.clamp(libdevice.rint(v / v_scale[:, None]), -127.0, 127.0).to(
            tl.int8
        )

        cache_offsets = (
            slot * stride_cb
            + h_idx * stride_ch
            + (cached_len + t_offsets[:, None]) * stride_cn
            + d_offsets[None, :]
        )
        tl.store(k_cache_ptr + cache_offsets, k_quantized, mask=in_t[:, None])
        tl.store(v_cache_ptr + cache_offsets, v_quantized, mask=in_t[:, None])
        scale_offsets = slot * stride_sb + h_idx * stride_sh + cached_len + t_offsets
        tl.store(k_scale_ptr + scale_offsets, k_scale, mask=in_t)
        tl.store(v_scale_ptr + scale_offsets, v_scale, mask=in_t)


def quantize_kv_prefill(
    k: torch.Tensor,  # bf16 [N, Hkv, D], all sequences packed along N
    v: torch.Tensor,  # bf16 [N, Hkv, D]
    cu_seqlens: torch.Tensor,  # int32 [B + 1]
    slot_ids: torch.Tensor,  # [B], sequence i is written to cache row slot_ids[i]
    cached_lens: torch.Tensor,  # [B], new tokens are written starting at this column
    cache: LayerKVCache,
):
    N, Hkv, D = k.shape
    assert k.stride(-1) == 1 and v.stride(-1) == 1, "kernel assumes contiguous head_dim"
    assert cache.k.stride() == cache.v.stride()
    assert cache.k_scale.stride() == cache.v_scale.stride()
    quantize_kv_prefill_kernel[(cu_seqlens.shape[0] - 1, Hkv)](
        k,
        v,
        cache.k,
        cache.v,
        cache.k_scale,
        cache.v_scale,
        cache.k_calibration,
        cu_seqlens,
        slot_ids,
        cached_lens,
        k.stride(1),  # head
        k.stride(0),  # token
        v.stride(1),
        v.stride(0),
        cache.k.stride(0),
        cache.k.stride(1),
        cache.k.stride(2),
        cache.k_scale.stride(0),
        cache.k_scale.stride(1),
        cache.k_calibration.stride(0),
        cache.k_calibration.stride(1),
        cu_seqlens.stride(0),
        D=D,  # type: ignore
        BLOCK_T=32,  # type: ignore
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bn, w, st in [
            (32, 4, 2),
            (32, 8, 2),
            (64, 8, 2),
            (128, 8, 2),
        ]
    ],
    key=["T_BUCKET"],
    cache_results=True,
)
@triton.jit
def prefill_attention_kernel(
    q_ptr,  # bf16 [N, Hq,  D], all sequences packed along N
    k_ptr,  # bf16 [N, Hkv, D]
    v_ptr,  # bf16 [N, Hkv, D]
    out_ptr,  # bf16 [N, Hq, D]
    seq_ids_ptr,  # [N] which sequence each packed token belongs to
    cached_lens_ptr,  # [B], for every sequence: how many tokens have already been prefilled? 0 for pure prefill
    tile_start_ptr,
    cu_seqlens_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_calibration_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_ids_ptr,
    stride_qh,
    stride_qn,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_on,
    stride_oh,
    stride_cachekb,
    stride_cachekh,
    stride_cachekn,
    stride_cachevb,
    stride_cachevh,
    stride_cachevn,
    stride_calibb,
    stride_calibh,
    stride_kscaleb,
    stride_kscaleh,
    stride_vscaleb,
    stride_vscaleh,
    qk_scale,
    N,
    T_BUCKET,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # query rows for a program
    BLOCK_N: tl.constexpr,  # keys per inner-loop step
):
    tile_idx, hq_idx = (
        tl.program_id(axis=0),
        tl.program_id(axis=1),
    )
    hkv_idx = hq_idx // GQA_RATIO

    row_start = tl.load(tile_start_ptr + tile_idx)
    seq = tl.load(seq_ids_ptr + row_start)
    slot = tl.load(slot_ids_ptr + seq)
    cached_len = tl.load(cached_lens_ptr + seq)
    seq_start = tl.load(cu_seqlens_ptr + seq)
    seq_end = tl.load(cu_seqlens_ptr + seq + 1)
    d_offsets = tl.arange(0, D)

    start_m = row_start + tl.arange(0, BLOCK_M)
    q_seq_mask = start_m < seq_end  # [BLOCK_M]
    Q = tl.load(
        q_ptr + stride_qh * hq_idx + start_m[:, None] * stride_qn + d_offsets[None, :],
        mask=q_seq_mask[:, None],
    )  # [BLOCK_M, D]
    k_calibration = tl.load(
        k_calibration_ptr + slot * stride_calibb + hkv_idx * stride_calibh + d_offsets
    )
    Qc = (Q * k_calibration).to(tl.bfloat16)

    max_old = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for i in range(0, cached_len, BLOCK_N):
        n_offsets = i + tl.arange(0, BLOCK_N)
        mask = n_offsets < cached_len  # [BLOCK_N]

        K = tl.load(
            k_cache_ptr
            + slot * stride_cachekb
            + hkv_idx * stride_cachekh
            + n_offsets[:, None] * stride_cachekn
            + d_offsets[None, :],
            mask=mask[:, None],
        ).to(
            tl.bfloat16
        )  # [BLOCK_N, D]

        k_scale = tl.load(
            k_scale_ptr + slot * stride_kscaleb + hkv_idx * stride_kscaleh + n_offsets,
            mask=mask,
            other=0.0,
        )

        v = tl.load(
            v_cache_ptr
            + slot * stride_cachevb
            + hkv_idx * stride_cachevh
            + n_offsets[:, None] * stride_cachevn
            + d_offsets[None, :],
            mask=mask[:, None],
            other=0.0,
        ).to(
            tl.bfloat16
        )  # (BLOCK_N, D)

        v_scale = tl.load(
            v_scale_ptr + slot * stride_vscaleb + hkv_idx * stride_vscaleh + n_offsets,
            mask=mask,
            other=0.0,
        )

        attn_scores = (
            tl.dot(Qc, tl.trans(K)) * k_scale[None, :] * qk_scale
        )  # [BLOCK_M, BLOCK_N]
        attn_scores = tl.where(mask, attn_scores, float("-inf"))

        max_score = tl.max(attn_scores, axis=1)
        max_new = tl.maximum(max_score, max_old)
        safe_max_new = tl.where(max_new == float("-inf"), 0.0, max_new)

        alpha = tl.exp2(max_old - safe_max_new)
        numerator = tl.exp2(attn_scores - safe_max_new[:, None])
        running_sum = running_sum * alpha + tl.sum(numerator, axis=1)

        weighted_sum = tl.dot((numerator * v_scale[None, :]).to(v.dtype), v)  # (BLOCK_M, D)
        accumulator = accumulator * alpha[:, None] + weighted_sum

        max_old = max_new

    for start_n in range(
        seq_start,
        tl.minimum(row_start + BLOCK_M, seq_end),
        BLOCK_N,
    ):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        # bounding by N (not seq_end) is fine: the causal mask below already hides
        # any key past this tile's last row, so keys of the next sequence never count
        in_bounds = n_offsets < N  # [BLOCK_N]
        k_tile_mask = in_bounds[:, None]  # [BLOCK_N, 1]

        K = tl.load(
            k_ptr
            + stride_kh * hkv_idx
            + n_offsets[:, None] * stride_kn
            + d_offsets[None, :],
            mask=k_tile_mask,
        )  # [BLOCK_N, D]
        mask = in_bounds[None, :] & (n_offsets[None, :] <= start_m[:, None])

        v = tl.load(
            v_ptr
            + hkv_idx * stride_vh
            + n_offsets[:, None] * stride_vn
            + d_offsets[None, :],
            mask=in_bounds[:, None],
            other=0.0,
        )  # (BLOCK_N, D)

        attn_scores = tl.dot(Q, tl.trans(K)) * qk_scale  # [BLOCK_M, BLOCK_N]
        attn_scores = tl.where(mask, attn_scores, float("-inf"))

        max_score = tl.max(attn_scores, axis=1)
        max_new = tl.maximum(max_score, max_old)
        safe_max_new = tl.where(max_new == float("-inf"), 0.0, max_new)

        alpha = tl.exp2(max_old - safe_max_new)
        numerator = tl.exp2(attn_scores - safe_max_new[:, None])
        running_sum = running_sum * alpha + tl.sum(numerator, axis=1)

        weighted_sum = tl.dot(numerator.to(v.dtype), v)  # (BLOCK_M, D)
        accumulator = accumulator * alpha[:, None] + weighted_sum

        max_old = max_new

    out = tl.where(running_sum[:, None] != 0, accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr
        + stride_oh * hq_idx
        + start_m[:, None] * stride_on
        + d_offsets[None, :],
        out,
        mask=q_seq_mask[:, None],
    )


def prefill_attention(
    q: torch.Tensor,  # bf16 [N, Hq, D], all sequences packed along N
    k: torch.Tensor,  # bf16 [N, Hkv, D]
    v: torch.Tensor,  # bf16 [N, Hkv, D]
    seq_ids: torch.Tensor,  # [N] which sequence each packed token belongs to
    cached_lens: torch.Tensor,  # [B] int32, tokens already in the cache per sequence
    cu_seqlens: torch.Tensor,  # [B + 1] int32
    tile_start: torch.Tensor,  # [num_tiles] int32, first packed row of each tile, built with PREFILL_BLOCK_M
    slot_ids: torch.Tensor,  # [B] int32, kv cache row of each sequence
    kv_cache: LayerKVCache,
    PREFILL_BLOCK_M: int,  # must be the same value tile_start was built with
) -> torch.Tensor:  # bf16 [N, Hq, D]
    N, Hq, D = q.shape
    Hkv = k.shape[1]
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    out = torch.empty(N, Hq, D, device=q.device, dtype=q.dtype)
    prefill_attention_kernel[(tile_start.shape[0], Hq)](
        q,
        k,
        v,
        out,
        seq_ids,
        cached_lens,
        tile_start,
        cu_seqlens,
        kv_cache.k,
        kv_cache.v,
        kv_cache.k_calibration,
        kv_cache.k_scale,
        kv_cache.v_scale,
        slot_ids,
        q.stride(1),  # head
        q.stride(0),  # token
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        out.stride(0),  # token
        out.stride(1),  # head
        kv_cache.k.stride(0),
        kv_cache.k.stride(1),
        kv_cache.k.stride(2),
        kv_cache.v.stride(0),
        kv_cache.v.stride(1),
        kv_cache.v.stride(2),
        kv_cache.k_calibration.stride(0),
        kv_cache.k_calibration.stride(1),
        kv_cache.k_scale.stride(0),
        kv_cache.k_scale.stride(1),
        kv_cache.v_scale.stride(0),
        kv_cache.v_scale.stride(1),
        D**-0.5
        * 1.4426950408889634,  # qk_scale, log2(e) folded in for exp2. saves a PTX instruction
        N,
        T_BUCKET=triton.next_power_of_2(N),
        GQA_RATIO=Hq // Hkv,
        D=D,
        BLOCK_M=PREFILL_BLOCK_M,
    )
    return out


@triton.jit
def silu_mul_kernel(
    gate_up_ptr,
    output_ptr,
    stride_in,
    stride_out,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # [N, 2D] -> [N, D], one program per (token, BLOCK_D columns)
    row, pid_d = tl.program_id(axis=0), tl.program_id(axis=1)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    gate = tl.load(gate_up_ptr + row * stride_in + d_offsets, mask=mask).to(tl.float32)
    data = tl.load(gate_up_ptr + row * stride_in + D + d_offsets, mask=mask).to(
        tl.float32
    )

    tl.store(
        output_ptr + row * stride_out + d_offsets,
        gate * tl.sigmoid(gate) * data,
        mask=mask,
    )


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:  # [N, 2D] -> [N, D]
    N, two_d = gate_up.shape
    D = two_d // 2
    out = torch.empty(N, D, device=gate_up.device, dtype=gate_up.dtype)
    BLOCK_D = 1024
    silu_mul_kernel[(N, triton.cdiv(D, BLOCK_D))](
        gate_up, out, gate_up.stride(0), out.stride(0), D, BLOCK_D=BLOCK_D  # type: ignore
    )
    return out


@triton.jit
def add_rms_norm_kernel(
    residual_stream_ptr,
    delta_ptr,
    norm_weight_ptr,
    new_residual_output_ptr,
    new_norm_output_ptr,
    eps,
    stride_r,
    stride_d,
    stride_new_residual_output,
    stride_new_norm_output,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # residual stream & delta are [N, D], one program per token
    row = tl.program_id(axis=0)
    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    residual_stream = tl.load(
        residual_stream_ptr + row * stride_r + d_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    delta = tl.load(delta_ptr + row * stride_d + d_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(norm_weight_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)

    residual_stream = (residual_stream + delta).to(
        new_residual_output_ptr.dtype.element_ty
    )
    r = residual_stream.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(r * r, axis=0) / D + eps)

    tl.store(
        new_residual_output_ptr + row * stride_new_residual_output + d_offsets,
        residual_stream,
        mask=mask,
    )
    tl.store(
        new_norm_output_ptr + row * stride_new_norm_output + d_offsets,
        residual_stream * inv_rms * weight,
        mask=mask,
    )


def add_rms_norm(
    residual: torch.Tensor, delta: torch.Tensor, rms_norm: torch.Tensor, eps: float
):  # residual, delta: [N, D]
    N, D = residual.shape
    assert residual.stride(-1) == 1 and delta.stride(-1) == 1
    new_residual = torch.empty_like(residual)
    new_normed_residual = torch.empty_like(residual)
    BLOCK_D = triton.next_power_of_2(D)
    add_rms_norm_kernel[(N,)](
        residual,
        delta,
        rms_norm,
        new_residual,
        new_normed_residual,
        eps,
        residual.stride(dim=0),
        delta.stride(dim=0),
        new_residual.stride(dim=0),
        new_normed_residual.stride(dim=0),
        D=D,  # type: ignore
        BLOCK_D=BLOCK_D,  # type: ignore
    )
    return new_residual, new_normed_residual


@triton.jit
def sampling_kernel(
    seed_ptr,
    logits_ptr,
    partial_max_ptr,
    partial_idx_ptr,
    tau,
    stride_lb,
    stride_pb,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # grid will be (B, cdiv(V, BLOCK_V))
    b_idx, pid_v = tl.program_id(axis=0), tl.program_id(axis=1)
    seed = tl.load(seed_ptr)

    v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < V

    logits = tl.load(
        logits_ptr + b_idx * stride_lb + v, mask=mask, other=float("-inf")
    ).to(tl.float32)

    u = tl.rand(seed, b_idx * V + v)  # ~U(0, 1)
    u = tl.clamp(u, 1e-10, 1.0 - 1e-7)
    logits = logits + tau * -tl.log(-tl.log(u))
    best, idx = tl.max(logits, axis=0, return_indices=True)

    # store the winning logit and its index, from the BLOCK_V we saw
    tl.store(partial_max_ptr + b_idx * stride_pb + pid_v, best)
    tl.store(partial_idx_ptr + b_idx * stride_pb + pid_v, pid_v * BLOCK_V + idx)


@triton.jit
def combine_partial_samples_kernel(
    partial_max_ptr,
    partial_idx_ptr,
    out_ptr,
    stride_pb,
    stride_ob,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # grid by just B, we wanna combine all partial results from above kernel
    b_idx = tl.program_id(axis=0)
    n = tl.arange(0, BLOCK_N)
    mask = n < NUM_BLOCKS

    m = tl.load(partial_max_ptr + b_idx * stride_pb + n, mask=mask, other=float("-inf"))
    _, slot = tl.max(m, axis=0, return_indices=True)

    token = tl.load(partial_idx_ptr + b_idx * stride_pb + slot)
    tl.store(out_ptr + b_idx * stride_ob, token)


def sample(logits: torch.Tensor, tau: float, seed: torch.Tensor) -> torch.Tensor:
    B, V = logits.shape
    assert logits.stride(-1) == 1, "kernel assumes contiguous vocab dim"
    BLOCK_V = 2048
    num_blocks = triton.cdiv(V, BLOCK_V)
    partial_max = torch.empty(B, num_blocks, device=logits.device, dtype=torch.float32)
    partial_idx = torch.empty(B, num_blocks, device=logits.device, dtype=torch.int64)
    out = torch.empty(B, 1, device=logits.device, dtype=torch.int64)
    sampling_kernel[(B, num_blocks)](
        seed,
        logits,
        partial_max,
        partial_idx,
        tau,
        logits.stride(0),
        partial_max.stride(0),
        V=V,  # type: ignore
        BLOCK_V=BLOCK_V,  # type: ignore
    )
    combine_partial_samples_kernel[(B,)](
        partial_max,
        partial_idx,
        out,
        partial_max.stride(0),
        out.stride(0),
        NUM_BLOCKS=num_blocks,  # type: ignore
        BLOCK_N=triton.next_power_of_2(num_blocks),  # type: ignore
    )
    seed.add_(1)
    return out
