# %%

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(123)

# %%
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],
    ]  # step
)

d_in = inputs.shape[1]
d_out = 2

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = inputs[1] @ W_query
key_2 = inputs[1] @ W_key
value_2 = inputs[1] @ W_value


print("query_2:", query_2.shape, "key_2:", key_2.shape, "value_2:", value_2.shape)
print(query_2, key_2, value_2, sep="\n\n")

# %%

query = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value
print("query:", query.shape, "keys:", keys.shape, "values:", values.shape)


# %%
attention_scores = query @ keys.T
d_k = keys.shape[-1]
attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
print(values)
print(attention_weights)

# %%

outputs = attention_weights @ values
print(outputs)


# %%
# Using a class
from torch import nn


class AttentionV1(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, dim_out))
        self.W_key = nn.Parameter(torch.rand(d_in, dim_out))
        self.W_value = nn.Parameter(torch.rand(d_in, dim_out))

    def forward(self, inputs):
        query = inputs @ self.W_query
        keys = inputs @ self.W_key
        values = inputs @ self.W_value

        attention_scores = query @ keys.T
        d_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
        outputs = attention_weights @ values
        return outputs


# %%
torch.manual_seed(123)

attention = AttentionV1(d_in, d_out)

print(attention(inputs))


# %%
class CausalAttention(nn.Module):
    def __init__(
        self, dim_in, dim_out, context_length, bias=False, dropout_prob=0.0, debug=False
    ):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob
        self.context_length = context_length
        self.dim_out = dim_out
        self.debug = debug

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs):
        batch_size, context_length, dim_in = inputs.shape

        query = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        attention_scores = query @ keys.transpose(-2, -1)

        attention_scores.masked_fill_(
            self.mask.bool()[:context_length, :context_length],  # pyright: ignore[reportCallIssue]
            float("-inf"),
        )
        if self.debug:
            print("Attention scores after mask:\n", attention_scores)

        d_k = keys.shape[-1]

        attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
        if self.debug:
            print("Attention weights:\n", attention_weights)

        attention_weights = self.dropout(attention_weights)
        if self.debug:
            print("Attention weights after dropout:\n", attention_weights)

        outputs = attention_weights @ values
        return outputs


# %%

batch = torch.stack([inputs, inputs])
print("Batch:\n", batch.shape)
ca = CausalAttention(d_in, d_out, inputs.shape[0])
print("Causal attention:\n", ca)
print(ca(batch))

# %%


class FakeParallelMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, inputs):
        outputs = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return outputs


# %%

torch.manual_seed(123)
context_length = batch.shape[1]

mha = FakeParallelMultiHeadAttention(d_in, d_out, context_length, 0.0, 2)
print("Multi-head attention:\n", mha)
print(mha(batch))


# %%
