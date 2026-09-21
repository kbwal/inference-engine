import torch
import torch.nn as nn
import math


class AttentionHead(nn.Module):
    def __init__(
        self,
        head_dim: int,
        W_Q: nn.Linear,
        W_K: nn.Linear,
        W_V: nn.Linear,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V

    def forward(self, x: torch.Tensor):
        # (seq_len, head_dim)
        q: torch.Tensor = self.W_Q(x)
        k: torch.Tensor = self.W_K(x)
        v: torch.Tensor = self.W_V(x)

        attention_scores = q @ k.T  # (seq_len, seq_len)
        attention_scores /= math.sqrt(self.head_dim)
        attention_scores = attention_scores.softmax(dim=-1)  # softmax along q

        out = attention_scores @ v  # (seq_len, head_dim)
        return out


class AttentionLayer(nn.Module):
    def __init__(
        self, model_dim: int, head_dim: int, num_q_heads: int, num_kv_heads: int
    ):
        super().__init__()
        self.heads: list[AttentionHead] = []
        self.O = nn.Linear(num_q_heads * head_dim, model_dim)
        self.W_Qs = [nn.Linear(model_dim, head_dim) for _ in range(num_q_heads)]
        self.W_Ks = [nn.Linear(model_dim, head_dim) for _ in range(num_kv_heads)]
        self.W_Vs = [nn.Linear(model_dim, head_dim) for _ in range(num_kv_heads)]
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
        return self.O(torch.cat(outs, dim=1))


T = 200
d_model = 2048
l = AttentionLayer(d_model, 512, 32, 8)
input = torch.randn((T, d_model))
print(l.forward(input).shape)


class Qwen3_4B(nn.Module):
    def __init__(self):
        self.model_dim = 2560
        self.mlp_intermediate_dim = 9728
        self.head_dim = 128
        self.num_q_heads = 32
        self.num_kv_heads = 8
        self.vocab_size = 151936
        self.num_layers = 36
        self.embed_tokens = nn.Linear(self.vocab_size, self.model_dim)
