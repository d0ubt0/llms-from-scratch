import math

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        head_dim: int,
        num_heads: int,
        context_length: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.wq = nn.Linear(dim_in, num_heads * head_dim)
        self.wk = nn.Linear(dim_in, num_heads * head_dim)
        self.wv = nn.Linear(dim_in, num_heads * head_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(num_heads * head_dim, dim_in)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, input: Tensor):
        batch_size, num_tokens, _ = input.shape

        querys = self.wq(input)
        keys = self.wk(input)
        values = self.wv(input)

        querys = querys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        att_scores: Tensor = querys @ keys.transpose(-2, -1)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        att_scores.masked_fill_(mask_bool, -torch.inf)

        att_weights = torch.softmax(att_scores / keys.shape[-1] ** 0.5, dim=-1)
        att_weights = self.dropout(att_weights)

        context_vec = (
            (att_weights @ values)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.num_heads * self.head_dim)
        )

        return self.out(context_vec)
