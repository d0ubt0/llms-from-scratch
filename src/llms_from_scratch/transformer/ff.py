from torch import nn


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, ff_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            # GPT-2 usa "gelu_new" (aproximación con tanh), no el GELU exacto.
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, in_dim),
        )

    def forward(self, inputs):
        return self.layer(inputs)
