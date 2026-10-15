from torch import nn

from llms_from_scratch.config import Config
from llms_from_scratch.transformer.attention import MultiHeadSelfAttention
from llms_from_scratch.transformer.ff import FeedForward


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            config.emb_dim,
            config.qkv_dim_per_head,
            config.att_heads,
            config.context_length,
            config.drop_rate,
        )

        self.att_norm = nn.LayerNorm(config.emb_dim)
        self.ff_norm = nn.LayerNorm(config.emb_dim)

        self.dropout = nn.Dropout(config.drop_rate)

        self.ff = FeedForward(config.emb_dim, config.ff_dim)

    def forward(self, input):
        norm_x = self.att_norm(input)
        out = self.attention(norm_x)

        out = self.dropout(out)

        input = input + out

        norm = self.ff_norm(input)
        out = self.ff(norm)

        out = self.dropout(out)

        return out + input
