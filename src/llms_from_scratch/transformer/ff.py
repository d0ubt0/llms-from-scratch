from torch import nn


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, ff_dim, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, in_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, inputs):
        return self.layer(inputs)
