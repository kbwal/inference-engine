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

    def forward(self, x: torch.Tensor, attention_mask=None):
        B, T, _ = x.shape
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = torch.repeat_interleave(
            k,
            self.num_q_heads // self.num_kv_heads,
            dim=1,
        )
        v = torch.repeat_interleave(
            v,
            self.num_q_heads // self.num_kv_heads,
            dim=1,
        )

        if attention_mask is not None:
            pos = attention_mask.long().cumsum(-1) - 1
            pos = pos.masked_fill(attention_mask == 0, 0)
        else:
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        q = self.rope(self.q_norm(q), pos)
        k = self.rope(self.k_norm(k), pos)
        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            valid_keys = attention_mask[:, None, None, :].bool()
            allowed = causal_mask[None, None, :, :] & valid_keys  # shape [B,1,T,T]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=False
            )  # attention shape is # [B,H,T,T] but the mask broadcasts over H
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(out)


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

    def forward(self, x: torch.Tensor, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
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

    def forward(self, x: torch.Tensor, attention_mask=None):
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.embed_tokens(x)
        for block in self.layers:
            x = block(x, attention_mask=attention_mask)
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

    def forward(self, x: torch.Tensor, attention_mask=None):  # returns logits
        assert x.ndim == 2, "expected input of shape (batch_size, seq_len)"
        x = self.model(x, attention_mask)
        if attention_mask is not None:
            B, T = attention_mask.shape
            positions = torch.arange(T, device=attention_mask.device)
            masked_positions = torch.where(attention_mask.bool(), positions, -1)
            last_indices = masked_positions.argmax(dim=-1)
            x = x[torch.arange(B, device=x.device), last_indices]
        else:
            x = x[:, -1, :]
        w = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(x, w)


def make_qwen_1_7():
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
        device="mps",
        dtype=torch.bfloat16,
    )
    return model
