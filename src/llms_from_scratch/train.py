"""Entrenamiento sencillo del LLM.

Uso local (CPU o CUDA):
    uv run python -m llms_from_scratch.train --data data/don_quijote.txt

Con un tokenizer BPE entrenado con scripts/train_tokenizer.py (ajusta
automáticamente `vocab_size` al del tokenizer):
    uv run python -m llms_from_scratch.train --data data/corpus_es.txt \
        --tokenizer checkpoints/tokenizer/tokenizer.json

En Colab (no reinstales torch: Colab ya trae el suyo optimizado):
    !bash scripts/train.sh --epochs 3          # GPU o CPU
    !bash scripts/train_tpu.sh --epochs 3      # TPU (requiere torch_xla)

En TPU, torch_xla debe ser de la misma versión que torch:
    %pip install "torch_xla==<version-de-torch>"
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llms_from_scratch.config import Config, get_config
from llms_from_scratch.data import LLMDataset
from llms_from_scratch.llm import LLM
from llms_from_scratch.tokenizer import (
    TextTokenizer,
    get_hf_tokenizer,
    get_tokenizer,
)

# Se rellenan solo cuando se entrena en TPU (torch_xla).
XM = None
PL = None

CONFIG_KEYS = (
    "emb_dim",
    "vocab_size",
    "ff_dim_multiplier",
    "layers",
    "att_heads",
    "context_length",
    "drop_rate",
    "qkv_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el LLM desde cero")

    parser.add_argument("--data", type=Path, default=Path("data/don_quijote.txt"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help=(
            "tokenizer.json de un BPE entrenado con scripts/train_tokenizer.py "
            "(por defecto se usa el tokenizer GPT-2 de tiktoken)"
        ),
    )
    parser.add_argument(
        "--config",
        choices=["micro", "t4", "small", "conventional"],
        default="micro",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Solapamiento entre ventanas (0 = igual al context_length)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fracción para validación (elige el mejor checkpoint)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fracción para test (solo se reporta, no elige checkpoint)",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xla"],
        default="auto",
        help="xla = TPU de Colab",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Imprime el estado cada N pasos (0 = desactivado)",
    )
    parser.add_argument(
        "--amp",
        choices=["auto", "off", "bf16", "fp16"],
        default="auto",
        help="Precisión mixta en CUDA (auto = bf16 si la GPU lo soporta)",
    )
    return parser.parse_args()


def is_tpu_environment() -> bool:
    return any(
        var in os.environ for var in ("PJRT_DEVICE", "COLAB_TPU_ADDR", "TPU_NAME")
    )


def setup_device(requested: str) -> tuple[torch.device, bool]:
    global XM, PL

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif is_tpu_environment():
            requested = "xla"
        else:
            requested = "cpu"

    if requested == "xla":
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        XM, PL = xm, pl
        device = xm.xla_device()
        print(f"Dispositivo: TPU ({device})")
        return device, True

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Pediste --device cuda, pero CUDA no está disponible.")

    device = torch.device(requested)
    print(f"Dispositivo: {device}")
    return device, False


def setup_amp(requested: str, device: torch.device) -> torch.dtype | None:
    """Precisión mixta solo en CUDA; CPU y XLA siguen en fp32."""
    if requested == "off" or device.type != "cuda":
        return None

    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return torch.bfloat16 if requested == "bf16" else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_text(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No existe {path}. Genera el dataset primero "
            "(por ejemplo con llms_from_scratch.cleaning_data)."
        )

    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise SystemExit(f"Formato no soportado: {suffix}. Usa .txt, .csv o .json.")

    column = "Contents" if "Contents" in df.columns else df.columns[0]
    messages = df[column].dropna().astype(str).str.strip()
    messages = messages[messages != ""]

    # <|endoftext|> separa los mensajes para que el modelo no los una.
    return "\n<|endoftext|>\n".join(messages)


def build_loaders(
    args: argparse.Namespace,
    tokenizer: TextTokenizer,
    config: Config,
    device: torch.device,
    use_xla: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, int]:
    text = load_text(args.data)
    if not text:
        raise SystemExit(f"{args.data} está vacío.")

    if (
        args.val_frac < 0
        or args.test_frac < 0
        or args.val_frac + args.test_frac >= 1
    ):
        raise SystemExit(
            "--val-frac y --test-frac deben ser >= 0 y sumar menos de 1."
        )

    print(f"Tokenizando {len(text):,} caracteres...")

    # Cortes por caracteres: train | val | test. Se guardan como índices para
    # no duplicar el corpus en memoria con slices.
    n = len(text)
    val_start = int(n * (1 - args.val_frac - args.test_frac))
    test_start = int(n * (1 - args.test_frac))

    stride = args.stride or config.context_length

    def make_dataset(start: int, end: int) -> LLMDataset | None:
        if end <= start:
            return None
        dataset = LLMDataset.from_range(
            text, start, end, tokenizer, config.context_length, stride
        )
        return dataset if len(dataset) > 0 else None

    def make_loader(
        dataset: LLMDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    train_dataset = make_dataset(0, val_start)
    if train_dataset is None:
        raise SystemExit(
            "El split de train es más corto que context_length. "
            "Usa más datos o baja la configuración."
        )

    val_dataset = make_dataset(val_start, test_start)
    test_dataset = make_dataset(test_start, n) if args.test_frac > 0 else None

    drop_last = len(train_dataset) >= args.batch_size
    steps_per_epoch = len(train_dataset) // args.batch_size if drop_last else 1

    train_loader = make_loader(train_dataset, shuffle=True, drop_last=drop_last)
    val_loader = (
        make_loader(val_dataset, shuffle=False, drop_last=False)
        if val_dataset is not None
        else None
    )
    test_loader = (
        make_loader(test_dataset, shuffle=False, drop_last=False)
        if test_dataset is not None
        else None
    )

    if val_loader is None:
        print("Aviso: el split de validación es muy pequeño; se omite.")
    if args.test_frac > 0 and test_loader is None:
        print("Aviso: el split de test es muy pequeño; se omite.")

    def token_count(dataset: LLMDataset | None) -> int:
        return len(dataset.tokens) if dataset is not None else 0

    print(
        f"Tokens: {token_count(train_dataset):,} train | "
        f"{token_count(val_dataset):,} val | "
        f"{token_count(test_dataset):,} test | "
        f"ventanas: {len(train_dataset):,} | "
        f"pasos por época: {steps_per_epoch:,}"
    )

    if use_xla:
        train_loader = PL.MpDeviceLoader(train_loader, device)
        if val_loader is not None:
            val_loader = PL.MpDeviceLoader(val_loader, device)
        if test_loader is not None:
            test_loader = PL.MpDeviceLoader(test_loader, device)

    return train_loader, val_loader, test_loader, steps_per_epoch


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


def config_to_dict(config: Config) -> dict:
    return {key: getattr(config, key) for key in CONFIG_KEYS}


def save_checkpoint(
    path: Path,
    model: LLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: Config,
    tokenizer_path: Path | None,
    epoch: int,
    step: int,
    val_loss: float,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config_to_dict(config),
        # generate.py lo usa para reconstruir los ids del vocabulario correcto.
        "tokenizer": None if tokenizer_path is None else str(tokenizer_path),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "best_val": best_val,
    }

    if XM is not None:
        XM.save(state, str(path))
    else:
        torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: LLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    step = checkpoint.get("step", 0)
    best_val = checkpoint.get("best_val", float("inf"))
    print(
        f"Reanudando desde {path} "
        f"(siguiente época: {start_epoch + 1}, paso {step:,})"
    )
    return start_epoch, step, best_val


@torch.no_grad()
def evaluate(
    model: LLM,
    loader: DataLoader,
    device: torch.device,
    use_xla: bool,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    total_loss = torch.zeros((), device=device)
    batches = 0

    with autocast_context(device, amp_dtype):
        for x, y in loader:
            if not use_xla:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

            total_loss += loss.detach()
            batches += 1

    model.train()
    return (total_loss / max(1, batches)).item()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device, use_xla = setup_device(args.device)

    if device.type == "cuda":
        # TF32 acelera las matmuls en GPUs Ampere o más nuevas de Colab.
        torch.set_float32_matmul_precision("high")

    amp_dtype = setup_amp(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    tokenizer = get_tokenizer("gpt2")
    config = get_config(args.config)

    if args.tokenizer is not None:
        tokenizer = get_hf_tokenizer(args.tokenizer)
        if tokenizer.vocab_size != config.vocab_size:
            print(
                f"Vocabulario: {config.vocab_size:,} (config {args.config}) -> "
                f"{tokenizer.vocab_size:,} (tokenizer propio)"
            )
        config.vocab_size = tokenizer.vocab_size

    print(f"Datos: {args.data}")
    print(
        f"Config: {args.config} | épocas: {args.epochs} | "
        f"batch: {args.batch_size} | lr: {args.lr} | "
        f"checkpoints: {args.out_dir}"
    )
    print(f"Precisión: {amp_dtype or 'fp32'}")
    print(config)

    train_loader, val_loader, test_loader, steps_per_epoch = build_loaders(
        args, tokenizer, config, device, use_xla
    )

    model = LLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(args.warmup, total_steps)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        start = time.perf_counter()

        for step, (x, y) in enumerate(train_loader):
            if not use_xla:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                # Desescala antes de recortar (no-op si AMP está desactivado).
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if use_xla:
                XM.optimizer_step(optimizer)
                XM.mark_step()
            else:
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            global_step += 1
            epoch_loss += loss.detach()

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                batch_loss = loss.item()
                mean_loss = (epoch_loss / (step + 1)).item()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - start
                print(
                    f"  época {epoch + 1}/{args.epochs} | "
                    f"paso {step + 1}/{steps_per_epoch} | "
                    f"batch loss {batch_loss:.4f} | "
                    f"media {mean_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
                )

        train_loss = (epoch_loss / max(1, steps_per_epoch)).item()

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, use_xla, amp_dtype)

        test_loss = float("nan")
        if test_loader is not None:
            test_loss = evaluate(model, test_loader, device, use_xla, amp_dtype)

        is_best = val_loader is not None and val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint(
            args.out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            config,
            args.tokenizer,
            epoch,
            global_step,
            val_loss,
            best_val,
        )
        if is_best:
            save_checkpoint(
                args.out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                args.tokenizer,
                epoch,
                global_step,
                val_loss,
                best_val,
            )

        metrics = [
            f"Época {epoch + 1}/{args.epochs}",
            f"train loss {train_loss:.4f}",
        ]
        if val_loader is not None:
            metrics.append(
                f"val loss {val_loss:.4f} | val ppl {perplexity(val_loss):.2f}"
            )
        if test_loader is not None:
            metrics.append(
                f"test loss {test_loss:.4f} | test ppl {perplexity(test_loss):.2f}"
            )
        print(" | ".join(metrics) + (" | mejor" if is_best else ""))

    print(f"Listo. Checkpoints en {args.out_dir} (last.pt y best.pt)")


if __name__ == "__main__":
    train(parse_args())
