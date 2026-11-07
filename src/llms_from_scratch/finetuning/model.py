"""Modelo de clasificación para el fine-tuning de SMS spam."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from llms_from_scratch.config import Config
from llms_from_scratch.llm import LLM
from llms_from_scratch.pretrained import load_pretrained_gpt2

TrainableScope = Literal[
    "head",
    "last_block",
    "last_2_blocks",
    "last_4_blocks",
    "full",
]

TRAINABLE_SCOPES: tuple[str, ...] = (
    "head",
    "last_block",
    "last_2_blocks",
    "last_4_blocks",
    "full",
)

SCOPE_TO_BLOCKS = {
    "head": 0,
    "last_block": 1,
    "last_2_blocks": 2,
    "last_4_blocks": 4,
}

GPT2_ALIASES = {
    "small": "gpt2",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
    "xl": "gpt2-xl",
    "gpt2": "gpt2",
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
}


def resolve_model_name(model: str) -> str:
    """Convierte ``small``/``medium``/... al nombre de Hugging Face."""
    key = model.strip().lower()
    if key not in GPT2_ALIASES:
        opciones = ", ".join(sorted(GPT2_ALIASES))
        raise ValueError(f"Modelo desconocido: {model!r}. Opciones: {opciones}")
    return GPT2_ALIASES[key]


class SpamClassifier(nn.Module):
    """GPT-2 + cabeza de clasificación binaria.

    El pooling usa el último token real de cada secuencia (ignorando el
    padding de la derecha). Con máscara causal, los tokens reales nunca
    atienden a ese padding, así que no hace falta una máscara de atención.
    """

    def __init__(self, backbone: LLM, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        emb_dim = backbone.token_embedding.embedding_dim

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(drop_rate),
            nn.Linear(emb_dim, 2),
        )

        # La proyección al vocabulario no se usa para clasificar y ocupa
        # decenas de millones de parámetros (50 257 x emb_dim).
        if hasattr(self.backbone, "out_projection"):
            del self.backbone.out_projection

    @classmethod
    def from_pretrained(
        cls,
        model: str = "small",
        *,
        device: str | torch.device = "cpu",
        drop_rate: float = 0.1,
    ) -> tuple["SpamClassifier", Config]:
        backbone, config = load_pretrained_gpt2(
            resolve_model_name(model), device=device
        )
        classifier = cls(backbone, drop_rate=drop_rate)
        # La cabeza se crea en CPU después del backbone: hay que moverla también.
        classifier.to(device)
        return classifier, config

    @property
    def emb_dim(self) -> int:
        return self.backbone.token_embedding.embedding_dim

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        hidden = self.backbone.forward_hidden(input_ids)
        last_index = (lengths - 1).clamp(min=0).to(hidden.device)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_index, last_index]
        return self.head(pooled)

    def predict_proba(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        logits = self(input_ids, lengths)
        return torch.softmax(logits.float(), dim=-1)


def configure_trainable(
    model: SpamClassifier,
    scope: TrainableScope | str = "last_2_blocks",
    unfreeze_last: int | None = None,
) -> int:
    """Congela el backbone y descongela solo la parte indicada.

    Devuelve el número de bloques transformer descongelados (sin contar la
    cabeza ni ``final_norm``).
    """
    blocks = list(model.backbone.transformers)
    num_blocks = len(blocks)

    if unfreeze_last is not None:
        n_blocks = int(unfreeze_last)
        if n_blocks < 0 or n_blocks > num_blocks:
            raise ValueError(
                f"unfreeze_last debe estar entre 0 y {num_blocks}, no {n_blocks}."
            )
        scope = "full" if n_blocks == num_blocks else "custom"
    else:
        if scope not in TRAINABLE_SCOPES:
            raise ValueError(
                f"scope desconocido: {scope!r}. Opciones: {', '.join(TRAINABLE_SCOPES)}"
            )
        n_blocks = SCOPE_TO_BLOCKS.get(scope, num_blocks)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # La cabeza siempre se entrena.
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    if scope == "full":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
    elif n_blocks > 0:
        for block in blocks[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in model.backbone.final_norm.parameters():
            parameter.requires_grad_(True)

    return n_blocks


def trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for _, p in trainable_parameters(model))


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def split_param_groups(
    model: SpamClassifier,
    *,
    lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Crea grupos con LR distinto para la cabeza y el backbone.

    LayerNorm y biases quedan sin weight decay.
    """
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []

    for name, parameter in trainable_parameters(model):
        if name.startswith("head.") or name.startswith("backbone.final_norm."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups: list[dict] = []
    for name, params, group_lr in (
        ("head", head_params, head_lr),
        ("backbone", backbone_params, lr),
    ):
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        if decay:
            groups.append(
                {
                    "name": f"{name}_decay",
                    "params": decay,
                    "lr": group_lr,
                    "weight_decay": weight_decay,
                }
            )
        if no_decay:
            groups.append(
                {
                    "name": f"{name}_no_decay",
                    "params": no_decay,
                    "lr": group_lr,
                    "weight_decay": 0.0,
                }
            )

    if not groups:
        raise ValueError("El modelo no tiene parámetros entrenables.")

    return groups
