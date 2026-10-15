from torch import Tensor, arange, nn

from llms_from_scratch.config import Config
from llms_from_scratch.transformer.model import Transformer


class LLM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.emb_dim)

        self.dropout = nn.Dropout(config.drop_rate)

        self.transformers = nn.Sequential(
            *(Transformer(config) for i in range(config.layers))
        )

        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_projection = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, seq_len = input.shape

        tokens_embeds = self.token_embedding(input)
        posit_embeds = self.positional_embedding(arange(seq_len, device=input.device))

        embeds = tokens_embeds + posit_embeds

        embeds = self.dropout(embeds)
        embeds = self.transformers(embeds)
        embeds = self.final_norm(embeds)

        return self.out_projection(embeds)
