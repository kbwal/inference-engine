import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import (
    LayerKVCache,
    decode_attention,
    fused_rope_kv_decode,
    quantize_kv_prefill,
    prefill_attention,
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
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: LayerKVCache,
        cache_pos: torch.Tensor,
        key_mask: torch.Tensor,
        pad_len: torch.Tensor,
    ):
        B, T, _ = x.shape  # note: x is newly generated tokens NOT in the kv cache
        q, k, v = self.qkv_proj(x).split(
            [
                self.head_dim * self.num_q_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if T == 1:
            q = fused_rope_kv_decode(
                q,
                k,
                v,
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                kv_cache,
                cache_pos,
                self.q_norm.eps,  # type: ignore
            )
            out = decode_attention(
                q, kv_cache, key_mask, cache_pos, self.head_dim**-0.5
            )
        else:
            q = self.rope(self.q_norm(q), cos=cos, sin=sin)
            k = self.rope(self.k_norm(k), cos=cos, sin=sin)
            quantize_kv_prefill(k, v, key_mask, kv_cache)
            out = prefill_attention(q, k, v, pad_len)

        out = out.contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLPLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        intermediate_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(
            model_dim, intermediate_dim, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            model_dim, intermediate_dim, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_dim, model_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor):
        # x does not include tokens in kv cache
        gate = F.silu(self.gate_proj(x))
        data = self.up_proj(x)
        return self.down_proj(gate * data)


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
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: LayerKVCache,
        cache_pos: torch.Tensor,
        key_mask: torch.Tensor,
        pad_len: torch.Tensor,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            cache_pos=cache_pos,
            key_mask=key_mask,
            pad_len=pad_len,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


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
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
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
        x: torch.Tensor,
        kv_cache: list[LayerKVCache],
        cache_pos: torch.Tensor,  # [T], cache slots to write into
        positions: torch.Tensor,  # [B, T], RoPE
        key_mask: torch.Tensor,  # [B, max_len] False for padding
    ):
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.embed_tokens(x)
        T = cache_pos.shape[0]

        angles = (
            positions[:, None, :, None]
            * self.get_buffer("inv_freq")[None, None, None, :]
        )
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        pad_len = T - key_mask[:, :T].sum(dim=1)

        for i, block in enumerate(self.layers):
            x = block(
                x,
                cos=cos,
                sin=sin,
                kv_cache=kv_cache[i],
                cache_pos=cache_pos,
                key_mask=key_mask,
                pad_len=pad_len,
            )
        x = self.norm(x)
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
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        cache_pos: torch.Tensor,  # [T], cache slots to write into
        positions: torch.Tensor,  # [B, T], RoPE
        key_mask: torch.Tensor,  # [B, max_len] False for padding
    ):  # returns logits
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.model(
            x,
            kv_cache=kv_cache,
            cache_pos=cache_pos,
            positions=positions,
            key_mask=key_mask,
        )
        x = x[:, -1, :]
        w = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(x, w)


def make_qwen_1_7(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
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
        device=device,
        dtype=dtype,
    )
    return model
