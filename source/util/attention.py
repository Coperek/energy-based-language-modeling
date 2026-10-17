import torch.nn as nn

from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


class FlashSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_theta=10_000, causal=True, window_size: int = -1,):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.rotary = RotaryEmbedding(
            self.head_dim,
            base=rope_theta,
            interleaved=True,
        )

        # (-1,-1) means global attention
        if window_size < 0:
            self.window = (-1, -1)
        else:
            self.window = (window_size, window_size)

    def calculate_query(self, x):
        B, S, _ = x.shape

        query = self.query(x)
        query = query.view(B, S, self.num_heads, self.head_dim)
        query = self.rotary(query)

        return query

    def calculate_key_value(self, x):
        B, S, _ = x.shape

        kv = self.key_value(x)
        kv = kv.view(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        k = self.rotary(k)

        return k, v

    def forward_qkv(self, q, k, v):

        out = flash_attn_func(
            q, k, v,
            causal=self.causal,
            window_size=self.window,
        )

        out = out.reshape(q.shape[0], q.shape[1], self.dim)

        return self.out(out)


    def forward(self, x):

        q = self.calculate_query(x)
        k, v = self.calculate_key_value(x)

        return self.forward_qkv(q, k, v)
