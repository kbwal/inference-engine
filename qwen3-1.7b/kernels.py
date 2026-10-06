import triton
import triton.language as tl
import torch


@triton.jit
def decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    key_mask_ptr,
    cache_pos_ptr,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_mb,
    scale,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # output is (B, H, 1, D), so index by (B, H)
    b_idx = tl.program_id(axis=0)  # batch
    h_idx = tl.program_id(axis=1)  # head
    kv_head_idx = h_idx // GQA_RATIO

    d_offsets = tl.arange(0, D)

    q_offsets = (b_idx * stride_qb + h_idx * stride_qh) + d_offsets
    q = tl.load(q_ptr + q_offsets)[None, :].to(tl.float32)  # (1, D)

    cache_pos = tl.load(cache_pos_ptr)  # [1]

    kv_bh_offsets = b_idx * stride_kb + kv_head_idx * stride_kh

    max_old = float("-inf")
    running_sum = 0.0
    accumulator = tl.zeros((D,), dtype=tl.float32)

    for start_n in range(0, cache_pos + 1, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        in_cache = n_offsets <= cache_pos  # [BLOCK_N]

        kv_nd_offsets = n_offsets[:, None] * stride_kn + d_offsets[None, :]
        k = tl.load(
            k_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        )  # (BLOCK_N, D)
        v = tl.load(
            v_ptr + kv_bh_offsets + kv_nd_offsets, mask=in_cache[:, None], other=0.0
        )  # (BLOCK_N, D)

        key_mask_offsets = b_idx * stride_mb + n_offsets
        key_mask = tl.load(key_mask_ptr + key_mask_offsets, mask=in_cache, other=False)[
            :, None
        ]  # [BLOCK_N, 1]

        mask = in_cache[:, None] & key_mask  # [BLOCK_N, 1]

        attn_scores = (
            tl.sum(k.to(tl.float32) * q, axis=1, keep_dims=True) * scale
        )  # (BLOCK_N, 1)
        attn_scores = tl.where(mask, attn_scores, float("-inf"))

        max_score = tl.max(attn_scores)
        max_new = tl.maximum(max_score, max_old)  # type: ignore
        safe_max_new = tl.where(max_new == float("-inf"), 0.0, max_new)
        alpha = tl.exp(max_old - safe_max_new)  # type: ignore

        numerator = tl.exp(attn_scores - safe_max_new)  # type: ignore
        running_sum = running_sum * alpha + tl.sum(numerator)

        weighted_sum = tl.sum(numerator * v.to(tl.float32), axis=0)
        accumulator = accumulator * alpha + weighted_sum
        max_old = max_new

    out = accumulator / running_sum
    out_offsets = (b_idx * stride_ob + h_idx * stride_oh) + d_offsets
    tl.store(output_ptr + out_offsets, out)


def decode_attention(
    q: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    key_mask: torch.Tensor,
    cache_pos: torch.Tensor,
    scale,
    BLOCK_N: int = 64,
):
    k_cache, v_cache = kv_cache
    B, Hq, T, D = q.shape
    assert T == 1, "somehow T != 1 in the custom decode attention kernel!"
    out = torch.empty(B, Hq, 1, D, device=q.device, dtype=q.dtype)
    decode_attention_kernel[(B, Hq)](
        q,
        k_cache,
        v_cache,
        out,
        key_mask,
        cache_pos,
        q.stride(dim=0),
        q.stride(dim=1),
        k_cache.stride(dim=0),
        k_cache.stride(dim=1),
        k_cache.stride(dim=2),
        out.stride(dim=0),
        out.stride(dim=1),
        key_mask.stride(dim=0),
        scale,
        GQA_RATIO=Hq // k_cache.shape[1],  # type: ignore
        D=D,  # type: ignore
        BLOCK_N=BLOCK_N,  # type: ignore
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
