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

    def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out

    def update_kv(
        self,
        past_kv: tuple[torch.Tensor, torch.Tensor],
        cache_size: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        T = new_k.size(2)
        past_kv[0][:, :, cache_size : cache_size + T] = new_k
        past_kv[1][:, :, cache_size : cache_size + T] = new_v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cache_size: int,
    ):
        B, T, _ = x.shape  # note: x is newly generated tokens NOT in the kv cache
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(self.q_norm(q), cos=cos, sin=sin)
        k = self.rope(self.k_norm(k), cos=cos, sin=sin)

        self.update_kv(kv_cache, cache_size, k, v)

        k = kv_cache[0][:, :, : cache_size + T]
        v = kv_cache[1][:, :, : cache_size + T]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

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
        mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cache_size: int,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            mask=mask,
            kv_cache=kv_cache,
            cache_size=cache_size,
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
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        cache_size: int,
        attention_mask: torch.Tensor | None = None,
    ):
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.embed_tokens(
            x[:, cache_size:]
        )  # everything in cache is sliced out during forward
        B, T, _ = x.shape

        if attention_mask is not None:
            pos = attention_mask.long().cumsum(-1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)[:, cache_size:]
            causal_mask = torch.ones(
                T, T + cache_size, dtype=torch.bool, device=x.device
            ).tril(diagonal=cache_size)
            valid_keys = attention_mask[:, None, None, :].bool()
            mask = causal_mask[None, None, :, :] & valid_keys  # [B,1,T,T]
        else:
            pos = (
                torch.arange(cache_size, cache_size + T, device=x.device)
                .unsqueeze(0)
                .expand(B, T)
            )
            mask = torch.tril(
                torch.ones(T, T + cache_size, device=x.device),
                diagonal=cache_size,
            ).bool()

        angles = (
            pos[:, None, :, None] * self.get_buffer("inv_freq")[None, None, None, :]
        )
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        for i, block in enumerate(self.layers):
            x = block(
                x,
                cos=cos,
                sin=sin,
                mask=mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                cache_size=cache_size,
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
        self.lm_head = nn.Linear(
            model_dim, vocab_size, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        cache_size: int,
        attention_mask: torch.Tensor | None = None,
    ):  # returns logits
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.model(
            x, attention_mask=attention_mask, kv_cache=kv_cache, cache_size=cache_size
        )
        x = x[:, -1, :]
        w = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(x, w)


def make_qwen_1_7(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
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
