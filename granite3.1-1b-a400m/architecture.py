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
        attention_multiplier: float = 0.015625,
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
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.attention_multiplier = attention_multiplier

    def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
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
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, _ = x.shape  # note: x is newly generated tokens NOT in the kv cache
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, cos=cos, sin=sin)
        k = self.rope(k, cos=cos, sin=sin)

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

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=self.attention_multiplier
        )  # granite replaces the usual 1/sqrt(head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(out), new_kv


class MOEParallelExperts(nn.Module):
    def __init__(self, num_experts, in_dim, out_dim, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_dim, in_dim, device=device, dtype=dtype)
        )


class MOERouter(nn.Module):
    def __init__(self, model_dim, num_experts, device, dtype):
        super().__init__()
        self.layer = nn.Linear(
            model_dim, num_experts, bias=False, device=device, dtype=dtype
        )


class MLPLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        intermediate_dim: int,
        num_experts: int,
        num_active_experts: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_active_experts = num_active_experts
        self.num_experts = num_experts
        self.router = MOERouter(model_dim, num_experts, device, dtype)
        self.input_linear = MOEParallelExperts(
            num_experts, model_dim, 2 * intermediate_dim, device, dtype
        )
        self.output_linear = MOEParallelExperts(
            num_experts, intermediate_dim, model_dim, device, dtype
        )

    def forward(self, x: torch.Tensor):
        # x does not include tokens in kv cache
        B, T, D = x.shape
        x = x.reshape(-1, D)  # [N, D]
        vals, idx = self.router.layer(x).topk(self.num_active_experts, dim=-1)
        expert_weights = vals.softmax(dim=-1).to(x.dtype)

        gates = torch.zeros(B * T, self.num_experts, device=x.device, dtype=x.dtype)
        gates = gates.scatter(
            1, idx, expert_weights
        )  # scatter expert weights into [N, num_experts], has num_active_experts non-zero per row
        gate, up = (x @ self.input_linear.weight.transpose(-1, -2)).chunk(
            2, dim=-1
        )  # [N, D] @ [num_experts, D, 2*intermediate_dim] -> [num_experts, N, 2*intermediate_dim] -> chunk
        y = (F.silu(gate) * up) @ self.output_linear.weight.transpose(
            -1, -2
        )  # [num_experts, N, D]
        return (y * gates.transpose(-1, -2).unsqueeze(-1)).sum(dim=0).view(B, T, D)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_experts: int,
        num_active_experts: int,
        attention_multiplier: float = 0.015625,
        residual_multiplier: float = 0.22,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.residual_multiplier = residual_multiplier
        self.self_attn = AttentionLayer(
            model_dim=model_dim,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            attention_multiplier=attention_multiplier,
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = nn.RMSNorm(
            model_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.block_sparse_moe = MLPLayer(
            model_dim=model_dim,
            intermediate_dim=mlp_intermediate_dim,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
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
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        attention_output, new_kv = self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            mask=mask,
            kv_cache=kv_cache,
        )
        x = x + attention_output * self.residual_multiplier
        x = (
            x
            + self.block_sparse_moe(self.post_attention_layernorm(x))
            * self.residual_multiplier
        )
        return x, new_kv


class Granite3_1Model(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        num_layers: int,
        num_experts: int,
        num_active_experts: int,
        base: int = 1500000,
        embedding_multiplier: float = 12.0,
        attention_multiplier: float = 0.015625,
        residual_multiplier: float = 0.22,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embedding_multiplier = embedding_multiplier
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
                    num_experts,
                    num_active_experts,
                    attention_multiplier=attention_multiplier,
                    residual_multiplier=residual_multiplier,
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
        attention_mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        cache_size = kv_cache[0][0].size(2) if kv_cache is not None else 0
        x = (
            self.embed_tokens(x[:, cache_size:]) * self.embedding_multiplier
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

        new_cache = []
        for i, block in enumerate(self.layers):
            x, layer_kv = block(
                x,
                cos=cos,
                sin=sin,
                mask=mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )
            new_cache.append(layer_kv)
        x = self.norm(x)
        return x, new_cache


class Granite3_1_1B_400M(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        num_layers: int,
        num_experts: int,
        num_active_experts: int,
        embedding_multiplier: float = 12.0,
        attention_multiplier: float = 0.015625,
        residual_multiplier: float = 0.22,
        logits_scaling: float = 6.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype
        self.logits_scaling = logits_scaling
        self.model = Granite3_1Model(
            model_dim=model_dim,
            mlp_intermediate_dim=mlp_intermediate_dim,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            embedding_multiplier=embedding_multiplier,
            attention_multiplier=attention_multiplier,
            residual_multiplier=residual_multiplier,
            device=device,
            dtype=dtype,
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
        w = self.model.embed_tokens.weight
        return F.linear(x, w) / self.logits_scaling, new_cache


def make_granite_1b_400m(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    model_dim = 1024
    mlp_intermediate_dim = 512
    head_dim = 64  # hidden_size / num_heads; config has no explicit head_dim
    num_q_heads = 16
    num_kv_heads = 8
    vocab_size = 49152
    num_layers = 24
    num_experts = 32
    num_active_experts = 8
    embedding_multiplier = 12.0
    attention_multiplier = 0.015625
    residual_multiplier = 0.22
    logits_scaling = 6.0
    model = Granite3_1_1B_400M(
        model_dim=model_dim,
        mlp_intermediate_dim=mlp_intermediate_dim,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_experts=num_experts,
        num_active_experts=num_active_experts,
        embedding_multiplier=embedding_multiplier,
        attention_multiplier=attention_multiplier,
        residual_multiplier=residual_multiplier,
        logits_scaling=logits_scaling,
        device=device,
        dtype=dtype,
    )
    return model
