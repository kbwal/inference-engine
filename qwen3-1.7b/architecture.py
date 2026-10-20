from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.cuda import nvtx
from kernels import (
    LayerKVCache,
    decode_attention,
    fused_rope_kv_decode,
    fused_norm_rope_prefill,
    quantize_kv_prefill,
    dequantize_kv_prefix,
    prefill_attention,
    silu_mul,
    add_rms_norm,
    linear,
)


@dataclass
class PrefixScratch:
    # bf16 copy of the prefilling sequences' cached tokens, rewritten by every layer,
    # so prefill attention doesn't unpack int8 a bajillion times
    k: torch.Tensor  # bf16 [Hkv, >= total_cached, D]
    v: torch.Tensor  # bf16 [Hkv, >= total_cached, D]
    cu_cached: (
        torch.Tensor
    )  # int32 [B + 1], entry b's prefix is rows [cu_cached[b], cu_cached[b+1]). decoders and unused entries have none


class AttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert (
            num_q_heads % num_kv_heads == 0
        ), "make sure the num_q_heads is a multiple of num_kv_heads!"
        self.o_proj = nn.Linear(
            num_q_heads * head_dim, model_dim, bias=False, device=device, dtype=dtype
        )
        self.qkv_proj = nn.Linear(
            model_dim,
            head_dim * (num_q_heads + 2 * num_kv_heads),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def forward(
        self,
        x: torch.Tensor,  # [N, model_dim], tokens NOT already in the kv cache
        # every sequence entry, numbered absolutely (no renumbering): the decoding ones first, one
        # token each, so decoding entry b's token is row b. then the prefilling ones
        seq_ids: (
            torch.Tensor | None
        ),  # [N], which entry each packed token belongs to. prefill only
        cu_seqlens: (
            torch.Tensor | None
        ),  # [B + 1], entry b is rows [cu_seqlens[b], cu_seqlens[b+1]). None means pure decode
        cos: torch.Tensor,  # [N, 1, head_dim // 2]
        sin: torch.Tensor,
        kv_cache: LayerKVCache,
        cached_lens: torch.Tensor,  # [B], tokens already in the kv cache per sequence
        slot_ids: torch.Tensor,  # [B], kv cache row of each sequence
        tile_start: (
            torch.Tensor | None
        ),  # [num_tiles], first row of each prefill attention program, -1 = no work. prefill only
        prefill_block_m: int,  # the BLOCK_M tile_start was built with
        prefix_scratch: PrefixScratch | None,  # prefill only
        num_decoding: torch.Tensor,  # int32 [1], how many entries are decoding. kernels read it to skip entries that aren't theirs
    ):
        N = x.shape[0]
        with nvtx.range("qkv_proj"):
            q, k, v = linear(x, self.qkv_proj.weight).split(
                [
                    self.head_dim * self.num_q_heads,
                    self.head_dim * self.num_kv_heads,
                    self.head_dim * self.num_kv_heads,
                ],
                dim=-1,
            )

        q = q.view(N, self.num_q_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        if cu_seqlens is None:
            # pure decode
            with nvtx.range("rope_kv_write"):
                q = fused_rope_kv_decode(
                    q,
                    k,
                    v,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos,
                    sin,
                    slot_ids,
                    kv_cache,
                    cached_lens,
                    num_decoding,
                    self.q_norm.eps,  # type: ignore
                )
            with nvtx.range("decode_attention"):
                out = decode_attention(
                    q,
                    kv_cache,
                    cached_lens,
                    slot_ids,
                    num_decoding,
                    self.head_dim**-0.5,
                )
        else:
            assert (
                seq_ids is not None
                and tile_start is not None
                and prefix_scratch is not None
            )
            # every kernel sees the whole batch and skips the entries that aren't its own, so
            # nothing here depends on how many sequences are decoding
            # norms + ropes the decode rows too, unused but cheap. the decode kernel does its own
            q_p, k_p = fused_norm_rope_prefill(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                self.q_norm.eps,  # type: ignore
            )
            quantize_kv_prefill(
                k_p,
                v,
                cu_seqlens,
                slot_ids,
                cached_lens,
                num_decoding,
                kv_cache,
            )
            dequantize_kv_prefix(
                kv_cache,
                prefix_scratch.cu_cached,
                slot_ids,
                prefix_scratch.k,
                prefix_scratch.v,
            )
            # [N, Hq, head_dim], fills the prefilling entries' rows
            out = prefill_attention(
                q_p,
                k_p,
                v,
                seq_ids,
                cu_seqlens,
                tile_start,
                prefix_scratch.k,
                prefix_scratch.v,
                prefix_scratch.cu_cached,
                prefill_block_m,
            )

            # decoding entry b's token is row b, so the first B rows hold every decoding token.
            # B comes from a shape, not from how many are decoding
            B = slot_ids.shape[0]
            q_d = fused_rope_kv_decode(
                q[:B],
                k[:B],
                v[:B],
                self.q_norm.weight,
                self.k_norm.weight,
                cos[:B],
                sin[:B],
                slot_ids,
                kv_cache,
                cached_lens,
                num_decoding,
                self.q_norm.eps,  # type: ignore
            )
            # fills rows [0, num_decoding) of the same buffer
            decode_attention(
                q_d,
                kv_cache,
                cached_lens,
                slot_ids,
                num_decoding,
                self.head_dim**-0.5,
                out=out,
            )

        with nvtx.range("o_proj"):
            # out is [N, Hq, head_dim]
            return linear(out.view(N, -1), self.o_proj.weight)


class MLPLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        intermediate_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            model_dim, 2 * intermediate_dim, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_dim, model_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        # x does not include tokens in kv cache
        gate_up = linear(x, self.gate_up_proj.weight)
        return linear(silu_mul(gate_up), self.down_proj.weight)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.self_attn = AttentionLayer(
            model_dim=model_dim,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = nn.RMSNorm(
            model_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.mlp = MLPLayer(
            model_dim=model_dim,
            intermediate_dim=mlp_intermediate_dim,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            model_dim, eps=1e-6, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,  # previous block's MLP
        residual: (
            torch.Tensor | None
        ),  # if there was a previous block, this is that residual
        seq_ids: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: LayerKVCache,
        cached_lens: torch.Tensor,
        slot_ids: torch.Tensor,
        tile_start: torch.Tensor | None,
        prefill_block_m: int,
        prefix_scratch: PrefixScratch | None,
        num_decoding: torch.Tensor,
    ):
        with nvtx.range("attn"):
            if residual is None:
                residual, normed = x, self.input_layernorm(x)
            else:
                residual, normed = add_rms_norm(
                    residual, x, self.input_layernorm.weight, self.input_layernorm.eps  # type: ignore
                )
            attn_out = self.self_attn(
                normed,
                seq_ids,
                cu_seqlens,
                cos=cos,
                sin=sin,
                kv_cache=kv_cache,
                cached_lens=cached_lens,
                slot_ids=slot_ids,
                tile_start=tile_start,
                prefill_block_m=prefill_block_m,
                prefix_scratch=prefix_scratch,
                num_decoding=num_decoding,
            )
        with nvtx.range("mlp"):
            residual, normed = add_rms_norm(
                residual,
                attn_out,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,  # type: ignore
            )
            return self.mlp(normed), residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        num_layers: int,
        base: int = 1000000,
        prefill_block_m: int = 128,  # query rows per prefill attention program
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.prefill_block_m = prefill_block_m
        self.embed_tokens = nn.Embedding(
            vocab_size, model_dim, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    model_dim,
                    mlp_intermediate_dim,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    device=device,
                    dtype=dtype,
                )
            )
        self.norm = nn.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        inv_freq_device = device if device != "meta" else "cuda"
        self.register_buffer(
            "inv_freq",
            base
            ** (
                -torch.arange(
                    0, head_dim, 2, dtype=torch.float32, device=inv_freq_device
                )
                / head_dim
            ),
            persistent=False,
        )

    def forward(
        self,
        # every input is a tensor the engine already filled, so nothing in here depends on python
        # values that change from step to step (what a cuda graph needs). see Engine.build_inputs
        x: torch.Tensor,  # [N] token ids. prefill: all chunks packed, decode: one per sequence
        kv_cache: list[LayerKVCache],
        num_decoding: torch.Tensor,  # int32 [1], how many entries are decoding (the first ones)
        cached_lens: torch.Tensor,  # [B], tokens already in the kv cache per entry
        slot_ids: torch.Tensor,  # [B], kv cache row of each entry
        # prefill only. None means pure decode
        positions: torch.Tensor | None = None,  # [N], each token's position in its sequence
        cu_seqlens: torch.Tensor | None = None,  # [B + 1], entry b is rows [cu_seqlens[b], cu_seqlens[b+1])
        seq_ids: torch.Tensor | None = None,  # [N], which entry each token belongs to
        tile_start: torch.Tensor | None = None,  # [num_tiles], first row of each prefill attention program, -1 = no work
        prefix_scratch: PrefixScratch | None = None,  # where the prefilling entries' cached prefixes get dequantized
    ):
        assert x.ndim == 1, "expected a flat [N] tensor of token ids"
        assert cu_seqlens is None or (
            positions is not None
            and seq_ids is not None
            and tile_start is not None
            and prefix_scratch is not None
        ), "a step with prefill needs all of its metadata"
        with nvtx.range("embed"):
            x = self.embed_tokens(x)  # [N, model_dim]

        with nvtx.range("metadata"):
            if cu_seqlens is None:
                # decode: the new token's position is the number of tokens before it
                positions = cached_lens
            assert positions is not None
            angles = (
                positions[:, None, None] * self.get_buffer("inv_freq")[None, None, :]
            )  # [N, 1, head_dim // 2], broadcasts over heads
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        residual = None
        for i, block in enumerate(self.layers):
            # same name for every layer, so nsys stats aggregates them into one row
            with nvtx.range("layer"):
                x, residual = block(
                    x,
                    residual,
                    seq_ids,
                    cu_seqlens,
                    cos=cos,
                    sin=sin,
                    kv_cache=kv_cache[i],
                    cached_lens=cached_lens,
                    slot_ids=slot_ids,
                    tile_start=tile_start,
                    prefill_block_m=self.prefill_block_m,
                    prefix_scratch=prefix_scratch,
                    num_decoding=num_decoding,
                )
        with nvtx.range("final_norm"):
            _, x = add_rms_norm(residual, x, self.norm.weight, self.norm.eps)  # type: ignore
        return x


class Qwen3_1_7B(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        num_layers: int,
        tie_word_embeddings: bool = True,
        prefill_block_m: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype
        self.model = Qwen3Model(
            model_dim=model_dim,
            mlp_intermediate_dim=mlp_intermediate_dim,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            num_layers=num_layers,
            prefill_block_m=prefill_block_m,
            device=device,
            dtype=dtype,
        )
        self.lm_head: nn.Linear | None = None
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(
                model_dim, vocab_size, bias=False, device=device, dtype=dtype
            )

    def forward(
        self,
        x: torch.Tensor,  # [N] token ids
        kv_cache: list[LayerKVCache],
        num_decoding: torch.Tensor,  # int32 [1], how many entries are decoding (the first ones)
        cached_lens: torch.Tensor,  # [B], tokens already cached per entry
        slot_ids: torch.Tensor,  # [B], kv cache row of each entry
        # prefill only, see Qwen3Model.forward
        positions: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        seq_ids: torch.Tensor | None = None,
        tile_start: torch.Tensor | None = None,
        prefix_scratch: PrefixScratch | None = None,
    ):  # returns logits [B, vocab]
        x = self.model(
            x,
            kv_cache=kv_cache,
            num_decoding=num_decoding,
            cached_lens=cached_lens,
            slot_ids=slot_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            seq_ids=seq_ids,
            tile_start=tile_start,
            prefix_scratch=prefix_scratch,
        )
        with nvtx.range("lm_head"):
            if cu_seqlens is not None:
                # each entry's last row. padding and unused entries give junk rows, never read
                x = x[cu_seqlens[1:] - 1]  # prefill
            # decode, already last
            w = (
                self.model.embed_tokens.weight
                if self.lm_head is None
                else self.lm_head.weight
            )
            return linear(x, w)


def make_qwen_1_7(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    prefill_block_m: int = 128,
):
    model_dim = 2048
    mlp_intermediate_dim = 6144
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    vocab_size = 151936
    num_layers = 28
    tie_word_embeddings = True
    model = Qwen3_1_7B(
        model_dim=model_dim,
        mlp_intermediate_dim=mlp_intermediate_dim,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
        tie_word_embeddings=tie_word_embeddings,
        prefill_block_m=prefill_block_m,
        device=device,
        dtype=dtype,
    )
    return model
