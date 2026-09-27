import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        device: str = "mps",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert (
            num_q_heads % num_kv_heads == 0
        ), "make sure the num_q_heads is a multiple of num_kv_heads!"
        self.o_proj = nn.Linear(
            num_q_heads * head_dim, model_dim, bias=False, device=device, dtype=dtype
        )
        self.q_proj = nn.Linear(
            model_dim, num_q_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            model_dim, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            model_dim, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def rope(self, x: torch.Tensor, positions: torch.Tensor, base=1000000):
        assert self.head_dim % 2 == 0, "head_dim must be even for rope to work!"
        inv_freq = base ** (
            -torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
            / self.head_dim
        )
        angles = positions[:, None, :, None] * inv_freq[None, None, None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out

    def update_kv(
        self,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_kv is None:
            return (new_k, new_v)
        combined_k = torch.cat((past_kv[0], new_k), dim=2)
        combined_v = torch.cat((past_kv[1], new_v), dim=2)
        return (combined_k, combined_v)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, _ = x.shape  # note: x is newly generated tokens NOT in the kv cache
        cache_size = kv_cache[0].size(2) if kv_cache else 0
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            pos = attention_mask.long().cumsum(-1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)[:, cache_size:]
        else:
            pos = (
                torch.arange(cache_size, cache_size + T, device=x.device)
                .unsqueeze(0)
                .expand(B, T)
            )

        q = self.rope(self.q_norm(q), pos)
        k = self.rope(self.k_norm(k), pos)

        new_kv = self.update_kv(kv_cache, k, v)

        k = torch.repeat_interleave(
            new_kv[0],
            self.num_q_heads // self.num_kv_heads,
            dim=1,
        )
        v = torch.repeat_interleave(
            new_kv[1],
            self.num_q_heads // self.num_kv_heads,
            dim=1,
        )

        if attention_mask is None:
            mask = torch.tril(
                torch.ones(T, T + cache_size, device=q.device),
                diagonal=cache_size,
            ).bool()
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False
            )
        else:
            causal_mask = torch.ones(
                T, T + cache_size, dtype=torch.bool, device=x.device
            ).tril(diagonal=cache_size)
            valid_keys = attention_mask[:, None, None, :].bool()
            allowed = causal_mask[None, None, :, :] & valid_keys  # [B,1,T,T]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=False
            )  # attention shape is [B,H,T,T] but the mask broadcasts over H
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(out), new_kv


class MLPLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        intermediate_dim: int,
        device: str = "mps",
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
        device: str = "mps",
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
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        attention_output, new_kv = self.self_attn(
            self.input_layernorm(x), attention_mask=attention_mask, kv_cache=kv_cache
        )
        x = x + attention_output
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


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
        device: str = "mps",
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        cache_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
        x = self.embed_tokens(
            x[:, cache_len:]
        )  # everything in cache is sliced out during forward
        new_cache = []
        for i, block in enumerate(self.layers):
            x, layer_kv = block(
                x,
                attention_mask=attention_mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )
            new_cache.append(layer_kv)
        x = self.norm(x)
        return x, new_cache


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
        device: str = "mps",
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
        self.lm_head = nn.Linear(
            model_dim, vocab_size, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):  # returns logits
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x, new_cache = self.model(x, attention_mask=attention_mask, kv_cache=kv_cache)
        x = x[:, -1, :]
        w = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(x, w), new_cache


def make_qwen_1_7(device: str = "mps", dtype: torch.dtype = torch.bfloat16):
    model_dim = 2048
    mlp_intermediate_dim = 6144
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    vocab_size = 151936
    num_layers = 28
    model = Qwen3_1_7B(
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
    return model
