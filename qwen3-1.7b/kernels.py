import triton
import triton.language as tl
import triton.testing
import torch
import torch.nn.functional as F
from triton.language.extra import libdevice
from dataclasses import dataclass
import functools


@dataclass
class LayerKVCache:
    k: torch.Tensor  # int8  [pool, Hkv, max_len, D]
    v: torch.Tensor  # int8  [pool, Hkv, max_len, D]
    k_scale: torch.Tensor  # fp32  [pool, Hkv, max_len]
    v_scale: torch.Tensor  # fp32  [pool, Hkv, max_len]
    k_calibration: torch.Tensor  # fp32  [pool, Hkv, 1, D]


@triton.jit
def compute_piece(
    cached_lens_ptr,
    num_decoding,
    BLOCK_B: tl.constexpr,
    MIN_PIECE: tl.constexpr,
    TARGET: tl.constexpr,
):
    # how many kv tokens one decode program handles this step, the same for every sequence.
    # sized from the whole batch's tokens so that ~TARGET programs per kv head have work.
    # the attention and combine kernels both call this, so they always agree on the splits
    b_offsets = tl.arange(0, BLOCK_B)
    cached_lens = tl.load(
        cached_lens_ptr + b_offsets, mask=b_offsets < num_decoding, other=-1
    )  # masked lanes become n_end = 0, so they don't count
    total = tl.sum(cached_lens + 1)
    piece = tl.maximum(MIN_PIECE, tl.cdiv(total, TARGET))
    return tl.cdiv(piece, MIN_PIECE) * MIN_PIECE  # round up to a multiple of MIN_PIECE


@triton.jit
def assign_work(
    cached_lens_ptr,
    num_decoding,
    work,  # this program's index in the work list
    BLOCK_B: tl.constexpr,
    MIN_PIECE: tl.constexpr,
    TARGET: tl.constexpr,
):
    # the work list is every sequence's pieces laid end to end: seq 0's pieces, then seq 1's, ...
    # find which (sequence, piece) entry `work` is. programs past the end get needed = 0
    piece = compute_piece(cached_lens_ptr, num_decoding, BLOCK_B, MIN_PIECE, TARGET)
    b_offsets = tl.arange(0, BLOCK_B)
    n_ends = (
        tl.load(cached_lens_ptr + b_offsets, mask=b_offsets < num_decoding, other=-1)
        + 1
    )
    splits = tl.cdiv(n_ends, piece)  # pieces per sequence, 0 for masked lanes
    ends = tl.cumsum(splits, axis=0)  # work index one past each sequence's last piece
    # sequences that end at or before `work` come before it, so their count is our sequence
    b_idx = tl.sum((ends <= work).to(tl.int32))
    valid = b_idx < num_decoding
    b_idx = tl.maximum(
        tl.minimum(b_idx, num_decoding - 1), 0
    )  # keep loads in bounds for invalid programs
    is_b = b_offsets == b_idx
    split_idx = work - tl.sum(
        tl.where(is_b, ends - splits, 0)
    )  # minus our sequence's first index
    needed = tl.where(valid, tl.sum(tl.where(is_b, splits, 0)), 0)
    return b_idx, split_idx, needed, piece


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bn in [32, 64, 128]
        for w in [4, 8]
        for st in [2, 3]
    ],
    key=["BLOCK_B", "max_len"],
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
    num_decoding_ptr,
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
    max_len,
    GQA_RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    MIN_PIECE: tl.constexpr,
    TARGET: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    num_decoding = tl.load(num_decoding_ptr)
    d_offsets = tl.arange(0, D)
    h_offsets = tl.arange(0, 16)
    h_valid = h_offsets < GQA_RATIO

    # axis 0 walks the work list of (sequence, piece) entries, axis 1 is the kv head
    b_idx, split_idx, needed, piece = assign_work(
        cached_lens_ptr, num_decoding, tl.program_id(axis=0), BLOCK_B, MIN_PIECE, TARGET
    )
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

    # this program covers tokens [split_start, split_end) of its sequence. past the end of the
    # work list needed is 0, so the range is empty and nothing gets stored below
    split_start = split_idx * piece
    split_end = tl.where(
        needed > 0, tl.minimum(split_start + piece, n_end), split_start
    )

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

    if needed == 1 and split_idx == 0:
        out = accumulator / running_sum[:, None]
        tl.store(
            output_ptr
            + b_idx * stride_ob
            + q_heads[:, None] * stride_oh
            + d_offsets[None, :],
            out,
            mask=h_valid[:, None],
        )
    elif needed > 1 and split_idx < needed:
        # we only saw a small slice, so we can't store the entire softmax in out
        HQ = tl.num_programs(axis=1) * GQA_RATIO
        rows = (b_idx * HQ + q_heads) * MAX_SPLITS + split_idx  # (16)
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
    cached_lens_ptr,
    num_decoding_ptr,
    stride_ob,
    stride_oh,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    MIN_PIECE: tl.constexpr,
    TARGET: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    num_decoding = tl.load(num_decoding_ptr)
    b_idx = tl.program_id(axis=0)
    if b_idx >= num_decoding:
        return
    q_head = tl.program_id(axis=1)
    s = tl.arange(0, MAX_SPLITS)
    d_offsets = tl.arange(0, D)
    rows = (b_idx * HQ + q_head) * MAX_SPLITS + s

    piece = compute_piece(cached_lens_ptr, num_decoding, BLOCK_B, MIN_PIECE, TARGET)
    cached_len = tl.load(cached_lens_ptr + b_idx)
    needed = tl.cdiv(cached_len + 1, piece)
    valid = s < needed

    maxs = tl.load(partial_max_ptr + rows, mask=valid, other=float("-inf"))  # (S)
    running_sum = tl.load(partial_running_sum_ptr + rows, mask=valid, other=0)  # (S)
    accumulator = tl.load(
        partial_accumulator_ptr + rows[:, None] * D + d_offsets[None, :],
        mask=valid[:, None],
        other=0,
    )  # (S, D)

    max_score = tl.max(maxs, axis=0)
    numerator = tl.where(maxs == float("-inf"), 0.0, tl.exp(maxs - max_score))  # (S)
    out = tl.sum(accumulator * numerator[:, None], axis=0) / tl.sum(
        running_sum * numerator, axis=0
    )  # type: ignore
    tl.store(
        output_ptr + b_idx * stride_ob + q_head * stride_oh + d_offsets,
        out,
        mask=needed > 1,
    )


@functools.cache
def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def decode_attention(
    q: torch.Tensor,  # [B, Hq, D], one new token per sequence
    kv_cache: LayerKVCache,
    cached_lens: torch.Tensor,  # [B], tokens already cached per sequence
    slot_ids: torch.Tensor,  # [B], kv cache row of each sequence
    num_decoding: torch.Tensor,  # a single number saying how many tokens are in decode
    scale,
    # [>= B, Hq, D]. if given, decoding entry b's output goes to row b of this instead of a new
    # tensor, so a mixed step's decode and prefill attention can fill one buffer
    out: torch.Tensor | None = None,
):  # -> [B, Hq, D], or out
    k_cache, v_cache = kv_cache.k, kv_cache.v
    assert (
        k_cache.stride() == v_cache.stride()
    ), "kernel uses one set of strides for k and v"
    assert kv_cache.k_scale.stride() == kv_cache.v_scale.stride()
    B, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    if out is None:
        out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)
    assert out.shape[0] >= B and out.shape[1:] == (Hq, D) and out.stride(-1) == 1

    PROGRAMS_PER_SM = 4  # a memory bound kernel needs several programs per SM in flight to hide latency
    MIN_PIECE = 64  # smallest kv piece per program. below this, writing partials costs more than the kv read

    # split the batch's kv tokens into pieces so that ~PROGRAMS_PER_SM programs per SM have work,
    # whatever the mix of lengths. TARGET is programs per kv head, and no sequence ever needs more
    # than TARGET pieces (piece >= total / TARGET), so partial buffers of that size always fit
    target = num_sms(q.device) * PROGRAMS_PER_SM // Hkv
    max_splits = triton.next_power_of_2(
        target
    )  # partial buffer stride, a power of 2 for tl.arange
    block_b = triton.next_power_of_2(B)
    partial_accumulator = torch.empty(
        B, Hq, max_splits, D, device=q.device, dtype=torch.float32
    )
    partial_maxes = torch.empty(B, Hq, max_splits, device=q.device, dtype=torch.float32)
    partial_running_sum = torch.empty(
        B, Hq, max_splits, device=q.device, dtype=torch.float32
    )

    # the work list holds at most target + B pieces: sum(cdiv(n_b, piece)) <= total / piece + B
    decode_attention_kernel[(target + B, Hkv)](
        q,
        k_cache,
        v_cache,
        out,
        kv_cache.k_scale,
        kv_cache.v_scale,
        cached_lens,
        num_decoding,
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
        max_len=triton.next_power_of_2(k_cache.shape[2]),
        GQA_RATIO=Hq // Hkv,
        D=D,
        BLOCK_B=block_b,
        MIN_PIECE=MIN_PIECE,
        TARGET=target,
        MAX_SPLITS=max_splits,
    )
    decode_attention_combine_kernel[(B, Hq)](
        partial_accumulator,
        partial_maxes,
        partial_running_sum,
        out,
        cached_lens,
        num_decoding,
        out.stride(dim=0),
        out.stride(dim=1),
        HQ=Hq,  # type: ignore
        D=D,  # type: ignore
        BLOCK_B=block_b,  # type: ignore
        MIN_PIECE=MIN_PIECE,  # type: ignore
        TARGET=target,  # type: ignore
        MAX_SPLITS=max_splits,  # type: ignore
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
    num_decoding_ptr,
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
    num_decoding = tl.load(num_decoding_ptr)
    b_idx = tl.program_id(axis=0)
    if b_idx >= num_decoding:
        return
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
    num_decoding: torch.Tensor,
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
        num_decoding,
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
def fused_norm_rope_prefill_kernel(
    q_ptr,  # bf16 [N, Hq, D], a view into qkv_proj's output, so rows are NOT contiguous
    k_ptr,  # bf16 [N, Hkv, D], same
    q_norm_ptr,  # [D]
    k_norm_ptr,  # [D]
    cos_ptr,  # bf16 [N, 1, D // 2], one row per packed token
    sin_ptr,
    q_out_ptr,  # bf16 [N, Hq, D], contiguous
    k_out_ptr,  # bf16 [N, Hkv, D], contiguous
    stride_qn,  # token stride of q. for qwen3-1.7b this is (16 + 2 * 8) * 128 = 4096, not 16 * 128
    stride_qh,
    stride_kn,
    stride_kh,
    stride_cosn,
    stride_sinn,
    stride_qon,
    stride_qoh,
    stride_kon,
    stride_koh,
    N,  # packed prefill tokens
    eps,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # tokens per program
):
    # prefill version of the norm + rope half of fused_rope_kv_decode_kernel
    # replaces norm + rope in the attention layer
    n_idx, h_idx = tl.program_id(0), tl.program_id(1)
    block_start = n_idx * BLOCK_T
    tokens = block_start + tl.arange(0, BLOCK_T)
    mask = (tokens < N)[:, None]
    half_d_offsets = tl.arange(0, D // 2)  # load half
    if h_idx < HQ:
        q_offset = tokens * stride_qn + h_idx * stride_qh

        x1 = tl.load(
            q_ptr + q_offset[:, None] + half_d_offsets[None, :], mask=mask, other=0
        ).to(
            tl.float32
        )  # [BLOCK_T, D // 2]
        x2 = tl.load(
            q_ptr + q_offset[:, None] + half_d_offsets[None, :] + D // 2,
            mask=mask,
            other=0,
        ).to(tl.float32)

        rstd = tl.rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)
        x1 = x1 * rstd[:, None] * tl.load(q_norm_ptr + half_d_offsets).to(tl.float32)
        x2 = (
            x2
            * rstd[:, None]
            * tl.load(q_norm_ptr + D // 2 + half_d_offsets).to(tl.float32)
        )
    else:
        k_offset = tokens * stride_kn + (h_idx - HQ) * stride_kh

        x1 = tl.load(
            k_ptr + k_offset[:, None] + half_d_offsets[None, :], mask=mask, other=0
        ).to(
            tl.float32
        )  # [D // 2]
        x2 = tl.load(
            k_ptr + k_offset[:, None] + half_d_offsets[None, :] + D // 2,
            mask=mask,
            other=0,
        ).to(tl.float32)

        rstd = tl.rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)
        x1 = x1 * rstd[:, None] * tl.load(k_norm_ptr + half_d_offsets).to(tl.float32)
        x2 = (
            x2
            * rstd[:, None]
            * tl.load(k_norm_ptr + D // 2 + half_d_offsets).to(tl.float32)
        )

    cos = tl.load(
        cos_ptr + tokens[:, None] * stride_cosn + half_d_offsets, mask=mask, other=0
    ).to(
        tl.float32
    )  # [D // 2]
    sin = tl.load(
        sin_ptr + tokens[:, None] * stride_sinn + half_d_offsets, mask=mask, other=0
    ).to(tl.float32)

    out1 = x1 * cos - x2 * sin  # [D // 2]
    out2 = x1 * sin + x2 * cos

    if h_idx < HQ:
        tl.store(
            q_out_ptr
            + tokens[:, None] * stride_qon
            + h_idx * stride_qoh
            + half_d_offsets,
            out1,
            mask=mask,
        )
        tl.store(
            q_out_ptr
            + tokens[:, None] * stride_qon
            + h_idx * stride_qoh
            + half_d_offsets
            + D // 2,
            out2,
            mask=mask,
        )
    else:
        tl.store(
            k_out_ptr
            + tokens[:, None] * stride_kon
            + (h_idx - HQ) * stride_koh
            + half_d_offsets,
            out1,
            mask=mask,
        )
        tl.store(
            k_out_ptr
            + tokens[:, None] * stride_kon
            + (h_idx - HQ) * stride_koh
            + half_d_offsets
            + D // 2,
            out2,
            mask=mask,
        )


def fused_norm_rope_prefill(
    q: torch.Tensor,  # bf16 [N, Hq, D], the prefill rows only: q[num_seqs_decoding:]
    k: torch.Tensor,  # bf16 [N, Hkv, D]
    q_norm_weight: torch.Tensor,  # [D]
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,  # [N, 1, D // 2]
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # normed + roped (q, k), contiguous
    N, Hq, D = q.shape
    Hkv = k.shape[1]
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "kernel assumes contiguous head_dim"
    assert cos.stride(-1) == 1 and sin.stride(-1) == 1
    q_out = torch.empty(N, Hq, D, device=q.device, dtype=q.dtype)
    k_out = torch.empty(N, Hkv, D, device=k.device, dtype=k.dtype)
    BLOCK_T = 32
    fused_norm_rope_prefill_kernel[(triton.cdiv(N, BLOCK_T), Hq + Hkv)](
        q,
        k,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        q_out,
        k_out,
        q.stride(0),  # token
        q.stride(1),  # head
        k.stride(0),
        k.stride(1),
        cos.stride(0),
        sin.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        N,
        eps,
        HQ=Hq,  # type: ignore
        D=D,  # type: ignore
        BLOCK_T=BLOCK_T,  # type: ignore
    )
    return q_out, k_out


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
    num_decoding_ptr,
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
    num_decoding = tl.load(num_decoding_ptr)
    b_idx = tl.program_id(axis=0)
    if b_idx < num_decoding:
        return
    h_idx = tl.program_id(axis=1)
    slot = tl.load(slot_ids_ptr + b_idx)
    d_offsets = tl.arange(0, D)

    seq_start = tl.load(cu_seqlens_ptr + b_idx * stride_cuseqb)
    # how many tokens am i writing?
    chunk_len = tl.load(cu_seqlens_ptr + (b_idx + 1) * stride_cuseqb) - seq_start
    if chunk_len == 0:
        return
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
    num_decoding: torch.Tensor,  # a single number that says how many tokens are in decode
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
        num_decoding,
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


@triton.jit
def dequantize_kv_prefix_kernel(
    k_cache_ptr,  # int8 [pool, Hkv, max_len, D]
    v_cache_ptr,  # int8 [pool, Hkv, max_len, D]
    k_scale_ptr,  # fp32 [pool, Hkv, max_len]
    v_scale_ptr,  # fp32 [pool, Hkv, max_len]
    k_calibration_ptr,  # fp32 [pool, Hkv, 1, D]
    k_out_ptr,  # bf16 [Hkv, total_cached, D], sequence b's prefix is rows [cu_cached[b], cu_cached[b+1])
    v_out_ptr,  # bf16 [Hkv, total_cached, D]
    cu_cached_ptr,  # int32 [B + 1]
    slot_ids_ptr,
    stride_cb,
    stride_ch,
    stride_cn,
    stride_sb,
    stride_sh,
    stride_calb,
    stride_calh,
    stride_oh,
    stride_on,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # the inverse of quantize_kv_prefill: k = int8 * k_scale * calibration, v = int8 * v_scale
    b_idx = tl.program_id(axis=0)
    h_idx = tl.program_id(axis=1)
    out_start = tl.load(cu_cached_ptr + b_idx)
    cached_len = tl.load(cu_cached_ptr + b_idx + 1) - out_start
    # the grid is sized for max_len, so most programs have no rows. leave before loading anything
    # else: with ~33 entries x 8 heads x max_len / 64 programs per layer, that adds up
    if tl.program_id(axis=2) * BLOCK_T >= cached_len:
        return
    slot = tl.load(slot_ids_ptr + b_idx)
    d_offsets = tl.arange(0, D)
    t_offsets = tl.program_id(axis=2) * BLOCK_T + tl.arange(0, BLOCK_T)
    in_t = t_offsets < cached_len

    cache_offsets = (
        slot * stride_cb
        + h_idx * stride_ch
        + t_offsets[:, None] * stride_cn
        + d_offsets[None, :]
    )
    scale_offsets = slot * stride_sb + h_idx * stride_sh + t_offsets
    out_offsets = (
        h_idx * stride_oh
        + (out_start + t_offsets)[:, None] * stride_on
        + d_offsets[None, :]
    )

    k = tl.load(k_cache_ptr + cache_offsets, mask=in_t[:, None], other=0).to(tl.float32)
    k_scale = tl.load(k_scale_ptr + scale_offsets, mask=in_t, other=0.0)
    calibration = tl.load(
        k_calibration_ptr + slot * stride_calb + h_idx * stride_calh + d_offsets
    )
    k = k * k_scale[:, None] * calibration[None, :]
    tl.store(k_out_ptr + out_offsets, k.to(tl.bfloat16), mask=in_t[:, None])

    v = tl.load(v_cache_ptr + cache_offsets, mask=in_t[:, None], other=0).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + scale_offsets, mask=in_t, other=0.0)
    tl.store(
        v_out_ptr + out_offsets,
        (v * v_scale[:, None]).to(tl.bfloat16),
        mask=in_t[:, None],
    )


def dequantize_kv_prefix(
    cache: LayerKVCache,
    cu_cached: torch.Tensor,  # int32 [B + 1], prefix lengths of every entry, cumulative. decoders and unused entries are 0
    slot_ids: torch.Tensor,  # [B], kv cache row of each entry
    k_out: torch.Tensor,  # bf16 [Hkv, >= total_cached, D], shared by every layer
    v_out: torch.Tensor,
):
    # always launches, with a grid that only depends on the cache's shape, so it can be captured.
    # an empty step just has every program fully masked
    B, (_, Hkv, max_len, D) = cu_cached.shape[0] - 1, cache.k.shape
    assert cache.k.stride() == cache.v.stride()
    assert cache.k_scale.stride() == cache.v_scale.stride()
    assert k_out.stride() == v_out.stride() and k_out.stride(-1) == 1
    BLOCK_T = 64
    dequantize_kv_prefix_kernel[(B, Hkv, triton.cdiv(max_len, BLOCK_T))](
        cache.k,
        cache.v,
        cache.k_scale,
        cache.v_scale,
        cache.k_calibration,
        k_out,
        v_out,
        cu_cached,
        slot_ids,
        cache.k.stride(0),
        cache.k.stride(1),
        cache.k.stride(2),
        cache.k_scale.stride(0),
        cache.k_scale.stride(1),
        cache.k_calibration.stride(0),
        cache.k_calibration.stride(1),
        k_out.stride(0),  # head
        k_out.stride(1),  # token
        D=D,  # type: ignore
        BLOCK_T=BLOCK_T,  # type: ignore
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
    tile_start_ptr,
    cu_seqlens_ptr,
    k_prefix_ptr,  # bf16 [Hkv, total_cached, D], from dequantize_kv_prefix
    v_prefix_ptr,  # bf16 [Hkv, total_cached, D]
    cu_cached_ptr,  # [B + 1], sequence b's cached prefix is rows [cu_cached[b], cu_cached[b+1])
    stride_qh,
    stride_qn,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_on,
    stride_oh,
    stride_prefixh,
    stride_prefixn,
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
    if row_start < 0:
        return
    seq = tl.load(seq_ids_ptr + row_start)
    prefix_start = tl.load(cu_cached_ptr + seq)
    cached_len = tl.load(cu_cached_ptr + seq + 1) - prefix_start
    seq_start = tl.load(cu_seqlens_ptr + seq)
    seq_end = tl.load(cu_seqlens_ptr + seq + 1)
    d_offsets = tl.arange(0, D)

    start_m = row_start + tl.arange(0, BLOCK_M)
    q_seq_mask = start_m < seq_end  # [BLOCK_M]
    Q = tl.load(
        q_ptr + stride_qh * hq_idx + start_m[:, None] * stride_qn + d_offsets[None, :],
        mask=q_seq_mask[:, None],
    )  # [BLOCK_M, D]

    max_old = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # the cached prefix, already dequantized to bf16 with its scales applied
    prefix_base = hkv_idx * stride_prefixh + prefix_start * stride_prefixn
    for i in range(0, cached_len, BLOCK_N):
        n_offsets = i + tl.arange(0, BLOCK_N)
        mask = n_offsets < cached_len  # [BLOCK_N]
        kv_offsets = (
            prefix_base + n_offsets[:, None] * stride_prefixn + d_offsets[None, :]
        )

        K = tl.load(k_prefix_ptr + kv_offsets, mask=mask[:, None])  # [BLOCK_N, D]
        v = tl.load(
            v_prefix_ptr + kv_offsets, mask=mask[:, None], other=0.0
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
    cu_seqlens: torch.Tensor,  # [B + 1] int32
    tile_start: torch.Tensor,  # [num_tiles] int32, first packed row of each tile, built with PREFILL_BLOCK_M
    k_prefix: torch.Tensor,  # bf16 [Hkv, total_cached, D], the cached prefixes from dequantize_kv_prefix
    v_prefix: torch.Tensor,  # bf16 [Hkv, total_cached, D]
    cu_cached: torch.Tensor,  # [B + 1] int32, where each sequence's prefix starts in k_prefix/v_prefix
    PREFILL_BLOCK_M: int,  # must be the same value tile_start was built with
) -> torch.Tensor:  # bf16 [N, Hq, D]
    N, Hq, D = q.shape
    Hkv = k.shape[1]
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert k_prefix.stride() == v_prefix.stride() and k_prefix.stride(-1) == 1
    out = torch.empty(N, Hq, D, device=q.device, dtype=q.dtype)
    prefill_attention_kernel[(tile_start.shape[0], Hq)](
        q,
        k,
        v,
        out,
        seq_ids,
        tile_start,
        cu_seqlens,
        k_prefix,
        v_prefix,
        cu_cached,
        q.stride(1),  # head
        q.stride(0),  # token
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        out.stride(0),  # token
        out.stride(1),  # head
        k_prefix.stride(0),  # head
        k_prefix.stride(1),  # token
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
        for bn in [32, 64, 128]
        for bk in [64, 128, 256]
        for w in [4, 8]
        for st in [2, 3, 4]
        if not (w == 8 and bn == 32)
    ],
    # runtime doesn't depend on the values in x or the weight, so tuning on any data is fine
    key=["N", "K", "BLOCK_M"],
    cache_results=True,
)
@triton.jit
def skinny_matmul_kernel(
    x_ptr,  # bf16 [M, K], activations
    weight_ptr,  # bf16 [N, K], weight rows, same layout as nn.Linear.weight
    y_ptr,  # bf16 [M, N], y = x @ weight.T
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,  # >= M, so every program covers all of M and reads its weights once
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mask = (tl.arange(0, BLOCK_M) < M)[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        w = tl.load(
            weight_ptr
            + ((pid * BLOCK_N + (tl.arange(0, BLOCK_N))) * stride_wn)[None, :]
            + (k_start + tl.arange(0, BLOCK_K))[:, None]
        )
        x = tl.load(
            x_ptr
            + (tl.arange(0, BLOCK_M) * stride_xm)[:, None]
            + (k_start + tl.arange(0, BLOCK_K))[None, :],
            mask=mask,
            other=0.0,
        )
        accumulator += tl.dot(x, w)

    tl.store(
        y_ptr
        + pid * BLOCK_N
        + (tl.arange(0, BLOCK_M) * stride_ym)[:, None]
        + tl.arange(0, BLOCK_N)[None, :],
        accumulator.to(tl.bfloat16),
        mask=mask,
    )


def skinny_matmul(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # [M, K] bf16, [N, K] bf16 -> [M, N] bf16
    # matmul for few rows (decode), where reading the weight is the whole cost. one program per
    # BLOCK_N weight rows covering all of M, so every weight byte is read once. cublas reads these
    # shapes at ~500-800 GB/s, this at ~750-880
    M, K = x.shape
    N = weight.shape[0]
    assert x.stride(-1) == 1 and weight.stride(-1) == 1
    assert M <= 64, "big M is compute bound, cublas is faster there"
    # every BLOCK_N / BLOCK_K in the configs divides these, so neither needs a mask
    assert N % 128 == 0 and K % 256 == 0
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    BLOCK_M = max(16, triton.next_power_of_2(M))  # tl.dot needs every dim >= 16
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    skinny_matmul_kernel[grid](
        x,
        weight,
        y,
        M,
        N,
        K,
        x.stride(0),
        weight.stride(0),
        y.stride(0),
        BLOCK_M=BLOCK_M,  # type: ignore
    )
    return y


# (N, K, M) -> whether skinny_matmul beat cublas for that weight shape and row count. filled by
# tune_linear before graph capture, anything it never measured uses cublas
use_skinny: dict[tuple[int, int, int], bool] = {}


def linear(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # [M, K] @ [N, K].T -> [M, N], with whichever measured faster
    if use_skinny.get((weight.shape[0], weight.shape[1], x.shape[0]), False):
        return skinny_matmul(x, weight)
    return F.linear(x, weight)


def tune_linear(weights: list[torch.Tensor], rows: list[int]) -> None:
    # times skinny_matmul against cublas for each weight shape at each row count and keeps the
    # faster one. do_bench clears l2 before every run, like the real model where each layer's
    # weights come from dram. a few shapes are within ~3%, so skinny_matmul needs a 2% win to switch
    for weight in weights:
        N, K = weight.shape
        for M in rows:
            if M > 64 or (N, K, M) in use_skinny:
                continue
            x = torch.randn(M, K, device=weight.device, dtype=weight.dtype)
            skinny = triton.testing.do_bench(lambda: skinny_matmul(x, weight), return_mode="median")
            cublas = triton.testing.do_bench(lambda: F.linear(x, weight), return_mode="median")
            use_skinny[(N, K, M)] = skinny < 0.98 * cublas
