import torch
import torch.nn as nn

from flash_attn import flash_attn_func


class FlashSelfAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            rope_theta=10_000,
            causal=True,
            window_size: int = -1,
            max_seq_len: int = 2048,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.max_seq_len = max_seq_len

        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"

        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # RoPE tables are precomputed once (fp32) as plain buffers, so nothing
        # mutates Python state during forward and torch.compile sees a pure function.
        # persistent=False keeps them out of the state_dict
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)  # (max_seq_len, head_dim/2)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

        # (-1,-1) means global attention
        if window_size < 0:
            self.window = (-1, -1)
        else:
            self.window = (window_size, window_size)

    def _rope(self, x, offset: int = 0):
        B, S, H, D = x.shape
        assert offset + S <= self.max_seq_len, f"position {offset + S} exceeds max_seq_len {self.max_seq_len}"

        cos = self.rope_cos[offset:offset + S][None, :, None, :]
        sin = self.rope_sin[offset:offset + S][None, :, None, :]

        xf = x.float().reshape(B, S, H, D // 2, 2)
        x1, x2 = xf.unbind(dim=-1)

        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return out.to(x.dtype)  # math in fp32, back to bf16/fp16 for flash-attn

    def calculate_query(self, x, offset: int = 0):
        B, S, _ = x.shape
        query = self.query(x).view(B, S, self.num_heads, self.head_dim)
        return self._rope(query, offset)

    def calculate_key_value(self, x, offset: int = 0):
        B, S, _ = x.shape
        kv = self.key_value(x).view(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        return self._rope(k, offset), v

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
