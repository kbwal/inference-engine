import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):
    def __init__(
        self,
        head_dim: int,
        W_Q: nn.Module,
        W_K: nn.Module,
        W_V: nn.Module,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, x: torch.Tensor):
        # (seq_len, head_dim)
        q: torch.Tensor = self.W_Q(x)
        k: torch.Tensor = self.W_K(x)
        v: torch.Tensor = self.W_V(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        attention_scores = q @ k.transpose(-1, -2)  # (seq_len, seq_len)
        mask = torch.tril(torch.ones(attention_scores.shape))
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_scores /= math.sqrt(self.head_dim)
        attention_scores = attention_scores.softmax(dim=-1)  # softmax along q

        out = attention_scores @ v  # (seq_len, head_dim)
        return out


class AttentionLayer(nn.Module):
    def __init__(
        self, model_dim: int, head_dim: int, num_q_heads: int, num_kv_heads: int
    ):
        super().__init__()
        self.heads = nn.ModuleList()
        self.O = nn.Linear(num_q_heads * head_dim, model_dim)
        self.W_Qs = nn.ModuleList()
        self.W_Ks = nn.ModuleList()
        self.W_Vs = nn.ModuleList()
        for _ in range(num_q_heads):
            self.W_Qs.append(nn.Linear(model_dim, head_dim))
        for _ in range(num_kv_heads):
            self.W_Ks.append(nn.Linear(model_dim, head_dim))
            self.W_Vs.append(nn.Linear(model_dim, head_dim))

        assert (
            num_q_heads % num_kv_heads == 0
        ), "make sure the num_q_heads is a multiple of num_kv_heads!"
        ratio = num_q_heads // num_kv_heads
        for i in range(num_q_heads):
            self.heads.append(
                AttentionHead(
                    head_dim=head_dim,
                    W_Q=self.W_Qs[i],
                    W_K=self.W_Ks[i // ratio],
                    W_V=self.W_Vs[i // ratio],
                )
            )

    def forward(self, x: torch.Tensor):
        outs = []
        for head in self.heads:
            outs.append(head.forward(x))
        return self.O(torch.cat(outs, dim=-1))


class MLPLayer(nn.Module):
    def __init__(self, model_dim: int, intermediate_dim: int):
        super().__init__()
        self.l1 = nn.Linear(model_dim, intermediate_dim)
        self.l2 = nn.Linear(model_dim, intermediate_dim)
        self.l3 = nn.Linear(intermediate_dim, model_dim)

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
