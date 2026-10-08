import torch.nn as nn

from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


class FlashSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_theta, window_size: int = -1,):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
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

    def forward(self, x):

        B, S, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)

        qkv_roped = self.rotary(qkv)

        q_roped, k_roped, v = qkv_roped.unbind(dim=2)

        out = flash_attn_func(
            q_roped, k_roped, v,
            causal=True,
            window_size=self.window,
        )

        out = out.reshape(B, S, self.dim)

        return self.out(out)