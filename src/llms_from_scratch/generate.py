"""Genera texto con un checkpoint entrenado.

Uso:
    uv run python -m llms_from_scratch.generate "En un lugar de la Mancha"

El tokenizer se toma del propio checkpoint (si se entrenó con
`--tokenizer`); si no, se usa el GPT-2 de tiktoken.

En TPU:
    PJRT_DEVICE=TPU uv run python -m llms_from_scratch.generate "..." --device xla
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from llms_from_scratch.config import Config
from llms_from_scratch.llm import LLM
from llms_from_scratch.pretrained import load_pretrained_gpt2
from llms_from_scratch.tokenizer import (
    EOS_TOKEN,
    TextTokenizer,
    get_hf_tokenizer,
    get_tokenizer,
)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera texto con un modelo entrenado"
    )
    parser.add_argument("prompt", help="Texto inicial a partir del cual generar")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta al .pt (por defecto checkpoints/best.pt o last.pt)",
    )
    source.add_argument(
        "--pretrained",
        nargs="?",
        const="gpt2",
        default=None,
        metavar="MODELO",
        help="Usa pesos oficiales de Hugging Face en vez de un checkpoint "
        "(por defecto gpt2; también gpt2-medium, gpt2-large, gpt2-xl)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer.json a usar (por defecto el guardado en el checkpoint; "
        "si no hay, el GPT-2 de tiktoken)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "xla"], default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Elige siempre el token más probable (ignora temperature/top-k)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad (por defecto, aleatoria en cada ejecución)",
    )
    return parser.parse_args()


def is_tpu_environment() -> bool:
    import os

    return any(
        var in os.environ for var in ("PJRT_DEVICE", "COLAB_TPU_ADDR", "TPU_NAME")
    )


def setup_device(requested: str) -> tuple[torch.device, object | None]:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif is_tpu_environment():
            requested = "xla"
        else:
            requested = "cpu"

    if requested == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device(), xm

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Pediste --device cuda, pero CUDA no está disponible.")

    return torch.device(requested), None


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise SystemExit(f"No existe el checkpoint {path}.")
        return path

    for name in ("best.pt", "last.pt"):
        candidate = DEFAULT_CHECKPOINT_DIR / name
        if candidate.exists():
            return candidate

    raise SystemExit(
        "No encontré checkpoints/best.pt ni checkpoints/last.pt. "
        "Entrena primero con ./scripts/train.sh"
    )


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[LLM, Config, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = Config(**checkpoint["config"])

    model = LLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, config, checkpoint


def resolve_tokenizer(path: Path | None, checkpoint: dict) -> TextTokenizer:
    """Usa el tokenizer con el que se entrenó (si está guardado) o GPT-2."""
    saved = checkpoint.get("tokenizer")
    if path is None and saved:
        path = Path(saved)
        if not path.exists():
            raise SystemExit(
                f"El checkpoint se entrenó con el tokenizer {saved}, que ya no "
                "existe. Pasá --tokenizer con la ruta correcta."
            )

    if path is None:
        return get_tokenizer("gpt2")

    tokenizer = get_hf_tokenizer(path)
    print(
        f"Tokenizer: {path} | vocabulario {tokenizer.vocab_size:,}",
        file=sys.stderr,
    )
    return tokenizer


@torch.no_grad()
def generate(
    model: LLM,
    tokenizer: TextTokenizer,
    prompt: str,
    config: Config,
    device: torch.device,
    xm: object | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    greedy: bool,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise SystemExit("El prompt está vacío.")

    if len(prompt_ids) > config.context_length:
        print(
            f"Aviso: el prompt tiene {len(prompt_ids)} tokens; "
            f"se usan los últimos {config.context_length}.",
            file=sys.stderr,
        )
        prompt_ids = prompt_ids[-config.context_length :]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = tokens.size(1)

    eos_id = tokenizer.special_token_id(EOS_TOKEN)

    for _ in range(max_new_tokens):
        logits = model(tokens[:, -config.context_length :])[:, -1, :]
        if xm is not None:
            xm.mark_step()

        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_values = torch.topk(logits, k).values
                logits = logits.masked_fill(logits < top_values[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

        tokens = torch.cat([tokens, next_token], dim=1)

    new_ids = tokens[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device, xm = setup_device(args.device)
    if args.pretrained is not None:
        if args.tokenizer is not None:
            raise SystemExit(
                "--tokenizer no aplica a --pretrained: los pesos oficiales de "
                "GPT-2 usan el tokenizer GPT-2 de tiktoken."
            )
        model, config = load_pretrained_gpt2(args.pretrained, device=device)
        tokenizer: TextTokenizer = get_tokenizer("gpt2")
        print(
            f"Pesos: {args.pretrained} (Hugging Face) | dispositivo {device}",
            file=sys.stderr,
        )
    else:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
        model, config, checkpoint = load_model(checkpoint_path, device)
        tokenizer = resolve_tokenizer(args.tokenizer, checkpoint)

        epoch = checkpoint.get("epoch", 0) + 1
        print(
            f"Checkpoint: {checkpoint_path} | época {epoch} | dispositivo {device}",
            file=sys.stderr,
        )

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        config=config,
        device=device,
        xm=xm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(text)


if __name__ == "__main__":
    main()
