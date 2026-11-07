"""Clasifica mensajes SMS como spam/ham con un checkpoint de fine-tuning.

Ejemplo::

    uv run python -m llms_from_scratch.finetuning.predict \
        --checkpoint checkpoints/spam/best.pt \
        "Congratulations! You won a prize. Call now!"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llms_from_scratch.finetuning.data import ID_TO_LABEL
from llms_from_scratch.finetuning.train import load_classifier
from llms_from_scratch.tokenizer import EOS_TOKEN, get_tokenizer

DEFAULT_CHECKPOINT = Path("checkpoints/spam/best.pt")


def predict_message(
    model,
    tokenizer,
    text: str,
    *,
    max_length: int = 128,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> dict:
    device = torch.device(device) if isinstance(device, str) else device
    token_ids = tokenizer.encode(text)[:max_length]

    if not token_ids:
        pad_token_id = tokenizer.special_token_id(EOS_TOKEN)
        token_ids = [50256 if pad_token_id is None else int(pad_token_id)]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        probabilities = model.predict_proba(input_ids, lengths)[0]

    spam_probability = float(probabilities[1].item())
    label_id = 1 if spam_probability >= threshold else 0

    return {
        "text": text,
        "label": ID_TO_LABEL[label_id],
        "label_id": label_id,
        "spam_probability": spam_probability,
        "ham_probability": float(probabilities[0].item()),
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica mensajes SMS como spam o ham"
    )
    parser.add_argument("text", nargs="+", help="Mensaje(s) a clasificar")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral de spam (por defecto el guardado en el checkpoint)",
    )
    parser.add_argument("--json", action="store_true", help="Salida en JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, _config, checkpoint = load_classifier(
        args.checkpoint, device=args.device
    )
    tokenizer = get_tokenizer("gpt2")

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    max_length = int(checkpoint.get("max_length", 128))
    device = next(model.parameters()).device

    predictions = [
        predict_message(
            model,
            tokenizer,
            text,
            max_length=max_length,
            threshold=threshold,
            device=device,
        )
        for text in args.text
    ]

    if args.json:
        print(json.dumps(predictions, indent=2, ensure_ascii=False))
        return

    for prediction in predictions:
        print(
            f"[{prediction['label']:>4}] "
            f"spam={prediction['spam_probability']:.4f} "
            f"(thr={prediction['threshold']:.2f}) "
            f"{prediction['text']}"
        )


if __name__ == "__main__":
    main()
