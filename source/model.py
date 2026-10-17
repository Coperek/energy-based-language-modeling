from util.attention import FlashSelfAttention

import torch.nn as nn
import torch

# TODO: varlen attention
class EnergyTransformerBlock(nn.Module):

    def __init__(self, dim, heads, rope_theta, context_window=-1, dropout=0.0):
        super().__init__()

        self.activation = nn.GELU(approximate="tanh")

        self.norm1 = nn.RMSNorm(dim)
        self.attention = FlashSelfAttention(dim, heads, rope_theta, window_size=context_window)
        self.dropout1 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.RMSNorm(dim)
        self.linear_up = nn.Linear(dim, dim * 4)
        self.linear_down = nn.Linear(dim * 4, dim)
        self.dropout2 = nn.Dropout(p=dropout / 2.0) if dropout > 0 else nn.Identity()

    def forward_training(self, first_set: torch.Tensor, second_set: torch.Tensor):

        assert self.training

        first_pass = self.norm1(first_set)
        second_pass = self.norm1(second_set)

        k_cache, v_cache = self.attention.calculate_key_value(first_pass)
        first_query = self.attention.calculate_query(first_pass)
        second_query = self.attention.calculate_query(second_pass)

        k_shifted = torch.cat([torch.zeros_like(k_cache[:, :1, :, :]), k_cache[:, :-1, :, :]], dim=1)
        v_shifted = torch.cat([torch.zeros_like(v_cache[:, :1, :, :]), v_cache[:, :-1, :, :]], dim=1)

        first_pass = self.dropout1(self.attention.forward_qkv(first_query, k_shifted, v_shifted))
        second_pass = self.dropout1(self.attention.forward_qkv(second_query, k_shifted, v_shifted))

        first_pass += first_set
        second_pass += second_set

        mlp_hidden_first_pass = self.linear_up(self.norm2(first_pass))
        mlp_hidden_second_pass = self.linear_up(self.norm2(second_pass))

        first_pass += self.dropout2(self.linear_down(self.activation(mlp_hidden_first_pass)))
        second_pass += self.dropout2(self.linear_down(self.activation(mlp_hidden_second_pass)))

        return first_pass, second_pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        # TODO: shift kv (or append empty token) like in training to avoid mismatch?

        attention_out = self.attention.forward(self.norm1(x))
        attention_out += x

        mlp_hidden = self.linear_up(self.norm2(attention_out))
        mlp_hidden = self.linear_down(self.activation(mlp_hidden))
        mlp_hidden += attention_out

        return mlp_hidden


class EnergyTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_blocks: int, vocab_size: int, rope_theta: int = 10_000, dropout: float = 0.0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)
        self.out = nn.Linear(dim, vocab_size)
        self.out.weight = self.embedding.weight # tied embeddings

        self.blocks = [
            EnergyTransformerBlock(dim, num_heads, rope_theta, dropout=dropout) for _ in range(num_blocks)
        ]

    def forward_training(self, first_set: torch.Tensor, second_set: torch.Tensor):

        assert self.training

        # TODO: produce second_set via augmentation

        out1 = self.embedding(first_set)
        out2 = self.embedding(second_set)

        for block in self.blocks:
            out1, out2 = block.forward_training(out1, out2)

        out1 = self.out(out1)
        out2 = self.out(out2)

        return out1, out2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        out = self.embedding(x)

        for block in self.blocks:
            out = block.forward(out)

        out = self.out(out)

        return nn.functional.softmax(out, dim=-1)
