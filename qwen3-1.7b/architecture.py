import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import (
    LayerKVCache,
    decode_attention,
    fused_rope_kv_decode,
    quantize_kv_prefill,
    prefill_attention,
    silu_mul,
    add_rms_norm,
)


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

    def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out

    def forward(
        self,
        x: torch.Tensor,  # [N, model_dim], tokens NOT already in the kv cache
        seq_ids: torch.Tensor | None,  # [N], prefill only
        cu_seqlens: torch.Tensor | None,  # [B + 1], prefill only (None means decode)
        cos: torch.Tensor,  # [N, 1, head_dim // 2]
        sin: torch.Tensor,
        kv_cache: LayerKVCache,
        cached_lens: torch.Tensor,  # [B], tokens already in the kv cache per sequence
        slot_ids: torch.Tensor,  # [B], kv cache row of each sequence
        tile_start: torch.Tensor | None,  # [num_tiles], prefill only
        prefill_block_m: int,  # the BLOCK_M tile_start was built with
    ):
        N = x.shape[0]
        q, k, v = self.qkv_proj(x).split(
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
            # decode
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
                self.q_norm.eps,  # type: ignore
            )
            out = decode_attention(
                q, kv_cache, cached_lens, slot_ids, self.head_dim**-0.5
            )
        else:
            assert seq_ids is not None and tile_start is not None
            q = self.rope(self.q_norm(q), cos=cos, sin=sin)
            k = self.rope(self.k_norm(k), cos=cos, sin=sin)
            quantize_kv_prefill(k, v, cu_seqlens, slot_ids, cached_lens, kv_cache)
            out = prefill_attention(
                q,
                k,
                v,
                seq_ids,
                cached_lens,
                cu_seqlens,
                tile_start,
                slot_ids,
                kv_cache,
                prefill_block_m,
            )

        return self.o_proj(out.view(N, -1))  # out is [N, Hq, head_dim]


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
        gate_up = self.gate_up_proj(x)
        return self.down_proj(silu_mul(gate_up))


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
    ):
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
        )
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
        x: torch.Tensor,  # [N] token ids. prefill: all prompts packed, decode: one per sequence
        kv_cache: list[LayerKVCache],
        cu_seqlens: torch.Tensor | None = None,  # [B + 1], prefill only
        cached_lens: (
            torch.Tensor | None
        ) = None,  # [B], tokens already in the kv cache per sequence. required for decode, defaults to 0 for prefill
        slot_ids: torch.Tensor | None = None,  # [B], kv cache row of each sequence
    ):
        assert x.ndim == 1, "expected a flat [N] tensor of token ids"
        assert (
            cu_seqlens is not None or cached_lens is not None
        ), "decode needs cached_lens"
        assert slot_ids is not None, "pass slot_ids for both prefill and decode"
        N = x.shape[0]
        x = self.embed_tokens(x)  # [N, model_dim]

        seq_ids = tile_start = None
        if cu_seqlens is not None:
            # prefill
            chunk_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            seq_ids = torch.repeat_interleave(
                input=torch.arange(chunk_lens.shape[0], device=x.device),
                repeats=chunk_lens,
                output_size=N,
            )
            if cached_lens is None:  # prefilling from scratch, nothing cached yet
                cached_lens = torch.zeros_like(chunk_lens)
            # a chunk continues after whatever is already cached
            positions = (
                cached_lens[seq_ids]
                + torch.arange(N, device=x.device)
                - cu_seqlens[seq_ids]
            )
            # first packed row of each prefill attention program. every sequence starts
            # on a fresh tile, so no tile holds rows from two sequences
            starts, row = [], 0
            for L in chunk_lens.tolist():
                starts.extend(range(row, row + L, self.prefill_block_m))
                row += L
            tile_start = torch.tensor(starts, device=x.device, dtype=torch.int32)
        else:
            # decode: the new token's position is the number of tokens before it
            assert cached_lens is not None
            positions = cached_lens

        angles = (
            positions[:, None, None] * self.get_buffer("inv_freq")[None, None, :]
        )  # [N, 1, head_dim // 2], broadcasts over heads
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        residual = None
        for i, block in enumerate(self.layers):
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
            )
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
        cu_seqlens: torch.Tensor | None = None,  # [B + 1], prefill only
        cached_lens: torch.Tensor | None = None,  # [B], tokens already cached. decode: required, prefill: default 0
        slot_ids: torch.Tensor | None = None,  # [B], kv cache row of each sequence
    ):  # returns logits [B, vocab]
        x = self.model(
            x,
            kv_cache=kv_cache,
            cu_seqlens=cu_seqlens,
            cached_lens=cached_lens,
            slot_ids=slot_ids,
        )
        if cu_seqlens is not None:
            x = x[cu_seqlens[1:] - 1]  # prefill
        # decode, already last
        w = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(x, w)


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
