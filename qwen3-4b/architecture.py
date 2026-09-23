import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLayer(nn.Module):
    def __init__(
        self, model_dim: int, head_dim: int, num_q_heads: int, num_kv_heads: int
    ):
        super().__init__()
        assert (
            num_q_heads % num_kv_heads == 0
        ), "make sure the num_q_heads is a multiple of num_kv_heads!"
        self.O = nn.Linear(num_q_heads * head_dim, model_dim, bias=False)
        self.W_Q = nn.Linear(model_dim, num_q_heads * head_dim, bias=False)
        self.W_K = nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
        self.W_V = nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def rope(self, x: torch.Tensor, positions: torch.Tensor, base=1000000):
        assert self.head_dim % 2 == 0, "head_dim must be even for rope to work!"
        inv_freq = base ** (-torch.arange(0, self.head_dim, 2) / self.head_dim)
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos(), angles.sin()

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        q: torch.Tensor = self.W_Q(x)
        k: torch.Tensor = self.W_K(x)
        v: torch.Tensor = self.W_V(x)

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

        pos = torch.arange(q.size(-2))
        q = self.rope(self.q_norm(q), pos)
        k = self.rope(self.k_norm(k), pos)

        attention_scores = q @ k.transpose(-1, -2)  # (B, num_heads, T, T)
        mask = torch.tril(torch.ones(attention_scores.shape))
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_scores /= math.sqrt(self.head_dim)
        attention_scores = attention_scores.softmax(dim=-1)  # softmax along q

        out = attention_scores @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.O(out)


class MLPLayer(nn.Module):
    def __init__(self, model_dim: int, intermediate_dim: int):
        super().__init__()
        self.l1 = nn.Linear(model_dim, intermediate_dim, bias=False)
        self.l2 = nn.Linear(model_dim, intermediate_dim, bias=False)
        self.l3 = nn.Linear(intermediate_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor):
        gate = F.silu(self.l1(x))
        data = self.l2(x)
        return self.l3(gate * data)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.attn = AttentionLayer(
            model_dim=model_dim,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )
        self.attn_norm = nn.RMSNorm(model_dim)
        self.mlp = MLPLayer(model_dim=model_dim, intermediate_dim=mlp_intermediate_dim)
        self.mlp_norm = nn.RMSNorm(model_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Qwen3_4B(nn.Module):
    def __init__(
        self,
        model_dim: int,
        mlp_intermediate_dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    model_dim, mlp_intermediate_dim, head_dim, num_q_heads, num_kv_heads
                )
            )
        self.final_norm = nn.RMSNorm(model_dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = F.linear(x, self.embedding.weight)
        return x


B = 4
T = 200
d_model = 2048
vocab_size = 100
l = Qwen3_4B(d_model, d_model * 4, 512, 32, 8, vocab_size, 10)
input = torch.randint(0, vocab_size - 1, (B, T))
print(l.forward(input).shape)
