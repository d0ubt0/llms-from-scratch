from typing import Literal


class Config:
    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        ff_dim_multiplier: int,
        layers: int,
        att_heads: int,
        context_length: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ) -> None:
        if emb_dim % att_heads != 0:
            raise ValueError("emb_dim debe ser divisible por att_heads")

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.ff_dim_multiplier = ff_dim_multiplier
        self.layers = layers
        self.att_heads = att_heads
        self.context_length = context_length
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias

        self.qkv_dim_per_head = emb_dim // att_heads
        self.ff_dim = emb_dim * ff_dim_multiplier

    def __str__(self) -> str:
        attrs = "\n".join(f"  {k}: {v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}(\n{attrs}\n)"


def get_config(
    model_size: Literal["micro", "small", "conventional"] = "small",
) -> Config:
    configs = {
        "micro": {
            "emb_dim": 128,
            "vocab_size": 50_257,
            "ff_dim_multiplier": 4,
            "layers": 4,
            "att_heads": 4,
            "context_length": 256,
            "drop_rate": 0.1,
            "qkv_bias": True,
        },
        "small": {
            "emb_dim": 768,
            "vocab_size": 50_257,
            "ff_dim_multiplier": 4,
            "layers": 12,
            "att_heads": 12,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
        },
        "conventional": {
            "emb_dim": 768,
            "vocab_size": 50_257,
            "ff_dim_multiplier": 4,
            "layers": 12,
            "att_heads": 12,
            "context_length": 1024,
            "drop_rate": 0.1,
            "qkv_bias": True,
        },
    }

    try:
        return Config(**configs[model_size])
    except KeyError:
        raise ValueError(
            f"Modelo desconocido: {model_size!r}. Opciones: {', '.join(configs)}"
        ) from None
