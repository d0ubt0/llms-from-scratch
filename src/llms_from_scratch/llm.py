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

    def forward_hidden(self, input: Tensor) -> Tensor:
        """Devuelve los estados ocultos antes de proyectar al vocabulario.

        Lo usa el clasificador de fine-tuning para quedarse con el último
        token en vez de calcular logits de todo el vocabulario.
        """
        _, seq_len = input.shape

        tokens_embeds = self.token_embedding(input)
        posit_embeds = self.positional_embedding(arange(seq_len, device=input.device))

        embeds = self.dropout(tokens_embeds + posit_embeds)

        embeds = self.transformers(embeds)
        return self.final_norm(embeds)

    def forward(self, input: Tensor) -> Tensor:
        return self.out_projection(self.forward_hidden(input))
