"""Carga los pesos oficiales de GPT-2 en nuestra implementación de LLM.

Los checkpoints de Hugging Face guardan las proyecciones como ``Conv1D``
(peso con forma ``(entrada, salida)``) y atan los pesos del ``lm_head`` a los
embeddings de tokens. Aquí traducimos ambos detalles a las capas
``nn.Linear`` de ``llms_from_scratch``.

Uso:
    from llms_from_scratch.pretrained import load_pretrained_gpt2

    model, config = load_pretrained_gpt2("gpt2")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from llms_from_scratch.config import Config
from llms_from_scratch.llm import LLM

# nn.LayerNorm usa 1e-5 por defecto, igual que todos los GPT-2.
GPT2_LAYER_NORM_EPS = 1e-5

# Nombre en Hugging Face -> nombre en nuestro Transformer.
_LINEAR_LAYERS = {
    "attn.c_proj": "attention.out",
    "mlp.c_fc": "ff.layer.0",
    "mlp.c_proj": "ff.layer.2",
}

_NORM_LAYERS = {
    "ln_1": "att_norm",
    "ln_2": "ff_norm",
}


def _hf_file(model_name: str, filename: str, cache_dir: str | Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_name,
            filename=filename,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )


def config_from_gpt2(raw: dict) -> Config:
    """Construye un ``Config`` a partir del ``config.json`` de GPT-2."""
    n_embd = int(raw["n_embd"])
    n_inner = raw.get("n_inner") or 4 * n_embd
    layer_norm_eps = float(raw.get("layer_norm_epsilon", GPT2_LAYER_NORM_EPS))

    if n_inner % n_embd != 0:
        raise ValueError(f"n_inner={n_inner} no es múltiplo de n_embd={n_embd}")
    if layer_norm_eps != GPT2_LAYER_NORM_EPS:
        raise ValueError("Solo se admite layer_norm_epsilon=1e-5")
    if raw.get("activation_function") != "gelu_new":
        raise ValueError(
            "Solo se admite activation_function='gelu_new' "
            f"(recibido: {raw.get('activation_function')!r})"
        )

    return Config(
        emb_dim=n_embd,
        vocab_size=int(raw["vocab_size"]),
        ff_dim_multiplier=n_inner // n_embd,
        layers=int(raw["n_layer"]),
        att_heads=int(raw["n_head"]),
        context_length=int(raw["n_positions"]),
        drop_rate=float(raw.get("resid_pdrop", 0.1)),
        qkv_bias=True,
    )


def _convert_block(
    prefix: str, name: str, tensor: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Traduce un peso de un bloque ``h.{i}`` al nombre de nuestro modelo."""
    if name == "attn.bias":
        # Máscara causal de Hugging Face; nuestro modelo la genera por capa.
        return {}

    param = name.rsplit(".", 1)[-1]

    if name.startswith("attn.c_attn."):
        # (768, 2304) -> tres matrices; el bias se parte en la dimensión 0.
        q, k, v = tensor.chunk(3, dim=0 if param == "bias" else 1)
        converted = {}
        for projection, part in zip(("q", "k", "v"), (q, k, v)):
            if param == "weight":
                part = part.t()
            converted[f"{prefix}.attention.w{projection}.{param}"] = (
                part.contiguous()
            )
        return converted

    if name.startswith(("attn.c_proj.", "mlp.c_fc.", "mlp.c_proj.")):
        target = _LINEAR_LAYERS[name.rsplit(".", 1)[0]]
        if param == "weight":
            # Conv1D guarda (entrada, salida); nn.Linear espera (salida, entrada).
            tensor = tensor.t().contiguous()
        return {f"{prefix}.{target}.{param}": tensor}

    if name.startswith(("ln_1.", "ln_2.")):
        norm, param = name.split(".")
        return {f"{prefix}.{_NORM_LAYERS[norm]}.{param}": tensor}

    raise ValueError(f"Peso inesperado en el checkpoint: {prefix}.{name}")


def convert_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convierte un ``state_dict`` de GPT-2 (HF) al de ``LLM``."""
    converted: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key.startswith("transformer."):
            key = key.removeprefix("transformer.")
        if key.startswith("lm_head."):
            # Está atado a los embeddings; se rellena al atar los pesos.
            continue

        if key == "wte.weight":
            converted["token_embedding.weight"] = tensor
        elif key == "wpe.weight":
            converted["positional_embedding.weight"] = tensor
        elif key.startswith("ln_f."):
            converted[f"final_norm.{key.removeprefix('ln_f.')}"] = tensor
        elif key.startswith("h."):
            _, layer, name = key.split(".", 2)
            converted.update(_convert_block(f"transformers.{layer}", name, tensor))
        else:
            raise ValueError(f"Peso inesperado en el checkpoint: {key!r}")

    # GPT-2 ata lm_head a wte; el checkpoint no siempre guarda lm_head.
    converted.setdefault(
        "out_projection.weight", converted["token_embedding.weight"]
    )

    return converted


def load_pretrained_gpt2(
    model_name: str = "gpt2",
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
) -> tuple[LLM, Config]:
    """Carga un GPT-2 oficial (``gpt2``, ``gpt2-medium``, ...) en ``LLM``.

    Descarga ``config.json`` y ``model.safetensors`` de Hugging Face la
    primera vez y los reutiliza desde la caché después.
    """
    config_file = _hf_file(model_name, "config.json", cache_dir)
    raw_config = json.loads(config_file.read_text())
    config = config_from_gpt2(raw_config)

    model = LLM(config)
    # GPT-2 ata la proyección de salida a los embeddings de tokens.
    model.out_projection.weight = model.token_embedding.weight

    state = convert_state_dict(
        load_file(_hf_file(model_name, "model.safetensors", cache_dir))
    )

    # Las máscaras causales son buffers propios: no vienen en el checkpoint.
    expected = {key for key in model.state_dict() if not key.endswith(".mask")}
    missing = expected - state.keys()
    unexpected = state.keys() - expected
    if missing or unexpected:
        raise RuntimeError(
            "Los pesos no encajan con el modelo. "
            f"Faltan: {sorted(missing)}. Sobran: {sorted(unexpected)}."
        )

    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), config
