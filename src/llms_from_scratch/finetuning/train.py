"""Fine-tuning de clasificación de SMS spam.

Ejemplos::

    # Configuración por defecto: GPT-2 small, últimos 2 bloques
    uv run python -m llms_from_scratch.finetuning.train

    # Solo la cabeza (linear probe) con GPT-2 medium
    uv run python -m llms_from_scratch.finetuning.train \
        --model medium --trainable head

    # Padding fijo y dataset balanceado
    uv run python -m llms_from_scratch.finetuning.train \
        --padding fixed --balance undersample
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.config import Config
from llms_from_scratch.finetuning.data import (
    DATA_FILE_PATH,
    ID_TO_LABEL,
    LABEL_TO_ID,
    load_spam_data,
    make_dataloaders,
    prepare_splits,
)
from llms_from_scratch.finetuning.model import (
    SpamClassifier,
    configure_trainable,
    count_total_parameters,
    count_trainable_parameters,
    resolve_model_name,
    split_param_groups,
    trainable_parameters,
)
from llms_from_scratch.tokenizer import EOS_TOKEN, get_tokenizer

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

DEFAULT_OUT_DIR = Path("checkpoints/spam")


@dataclass
class TrainingConfig:
    model: str = "small"
    trainable: str = "last_2_blocks"
    unfreeze_last: int | None = None
    padding: str = "dynamic"
    balance: str = "none"
    max_length: int = 128
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    val_size: float = 0.1
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    drop_rate: float = 0.1
    patience: int = 2
    min_delta: float = 0.0
    log_every: int = 50
    smoke: int = 0
    cache_features: bool = True
    out_dir: str = str(DEFAULT_OUT_DIR)
    data_file: str = str(DATA_FILE_PATH)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Pediste --device cuda, pero CUDA no está disponible.")
    if requested not in {"cpu", "cuda"}:
        raise SystemExit(f"Dispositivo no soportado: {requested!r}. Usa auto, cpu o cuda.")

    return torch.device(requested)


def setup_amp(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


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
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def binary_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().long()
    preds = (probs >= threshold).long()

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def tune_threshold(
    probs: torch.Tensor,
    labels: torch.Tensor,
    start: float = 0.05,
    end: float = 0.95,
    steps: int = 91,
) -> tuple[float, dict]:
    """Elige el umbral que maximiza F1 de spam en validación."""
    best_threshold = 0.5
    best_metrics = binary_metrics(probs, labels, best_threshold)
    best_f1 = best_metrics["f1"]

    for threshold in torch.linspace(start, end, steps).tolist():
        metrics = binary_metrics(probs, labels, threshold)
        if metrics["f1"] > best_f1 + 1e-12:
            best_threshold, best_f1, best_metrics = threshold, metrics["f1"], metrics

    return best_threshold, best_metrics


def class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label_id"].value_counts()
    n = len(train_df)
    weights = [
        n / (2.0 * max(1, int(counts.get(label_id, 0))))
        for label_id in range(len(LABEL_TO_ID))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def config_to_dict(config: Config) -> dict:
    return {key: getattr(config, key) for key in CONFIG_KEYS}


@torch.no_grad()
def evaluate_loader(
    model: SpamClassifier,
    loader,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for input_ids, lengths, targets in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        logits = model(input_ids, lengths)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * len(targets)
        total_examples += len(targets)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
        labels.append(targets.cpu())

    model.train()
    return (
        total_loss / max(1, total_examples),
        torch.cat(probs),
        torch.cat(labels),
    )


@torch.no_grad()
def evaluate_features(
    model: SpamClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 256,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.head.eval()
    total_loss = 0.0
    probs: list[torch.Tensor] = []

    for start in range(0, len(features), batch_size):
        batch_features = features[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        logits = model.head(batch_features)
        total_loss += F.cross_entropy(logits, batch_labels).item() * len(batch_labels)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())

    model.head.train()
    return total_loss / max(1, len(features)), torch.cat(probs), labels.cpu()


@torch.no_grad()
def extract_features(
    model: SpamClassifier,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula el hidden del último token una sola vez (para linear probe)."""
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for input_ids, lengths, targets in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)

        hidden = model.backbone.forward_hidden(input_ids)
        last_index = (lengths - 1).clamp(min=0)
        batch_index = torch.arange(hidden.size(0), device=device)
        features.append(hidden[batch_index, last_index].cpu())
        labels.append(targets)

    model.train()
    return torch.cat(features), torch.cat(labels)


def train_epoch(
    model: SpamClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    grad_clip: float,
    log_every: int,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    start = time.perf_counter()

    for step, (input_ids, lengths, targets) in enumerate(loader):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            logits = model(input_ids, lengths)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(targets)
        total_examples += len(targets)

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"  época {epoch + 1}/{epochs} | paso {step + 1}/{len(loader)} | "
                f"loss {loss.item():.4f} | "
                f"media {total_loss / max(1, total_examples):.4f} | {elapsed:.1f}s"
            )

    return total_loss / max(1, total_examples)


def train_head_epoch(
    model: SpamClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    batch_size: int,
    grad_clip: float,
) -> float:
    model.head.train()
    n = len(features)
    permutation = torch.randperm(n, device=features.device)
    total_loss = 0.0

    for start in range(0, n, batch_size):
        index = permutation[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = model.head(features[index])
        loss = criterion(logits, labels[index])
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(index)

    return total_loss / max(1, n)


def save_checkpoint(
    path: Path,
    model: SpamClassifier,
    config: Config,
    cfg: TrainingConfig,
    threshold: float,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    trainable_names = {name for name, _ in trainable_parameters(model)}
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }

    payload = {
        "model": state,
        "state_format": "trainable_only",
        "base_model": resolve_model_name(cfg.model),
        "model_alias": cfg.model,
        "config": config_to_dict(config),
        "trainable": cfg.trainable,
        "unfreeze_last": cfg.unfreeze_last,
        "max_length": cfg.max_length,
        "padding": cfg.padding,
        "balance": cfg.balance,
        "label_map": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "threshold": threshold,
        "drop_rate": cfg.drop_rate,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_classifier(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[SpamClassifier, Config, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_device = (
        resolve_device(device) if isinstance(device, str) else device
    )

    model, config = SpamClassifier.from_pretrained(
        checkpoint["base_model"],
        device=resolved_device,
        drop_rate=float(checkpoint.get("drop_rate", 0.1)),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise RuntimeError(
            f"El checkpoint tiene parámetros que no encajan: {sorted(unexpected)}"
        )

    # Los parámetros congelados vienen del GPT-2 original; que falten es
    # esperado en los checkpoints que solo guardan la parte entrenable.
    _ = missing
    model.eval()
    return model, config, checkpoint


def _limit_split(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def run_training(cfg: TrainingConfig, *, verbose: bool = True) -> dict:
    if cfg.epochs <= 0:
        raise ValueError("epochs debe ser mayor que 0.")
    if cfg.max_length <= 0:
        raise ValueError("max_length debe ser mayor que 0.")
    if cfg.padding not in {"dynamic", "fixed"}:
        raise ValueError("padding debe ser 'dynamic' o 'fixed'.")
    if cfg.balance not in {"none", "undersample"}:
        raise ValueError("balance debe ser 'none' o 'undersample'.")
    if not 0 <= cfg.val_size < 1:
        raise ValueError("val_size debe estar en [0, 1).")

    total_start = time.perf_counter()
    out_dir = Path(cfg.out_dir)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp_dtype = setup_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    if verbose:
        print("=" * 72)
        print(
            f"Fine-tuning spam | model={cfg.model} | trainable={cfg.trainable} | "
            f"padding={cfg.padding} | balance={cfg.balance}"
        )
        print(f"Dispositivo: {device} | precisión: {amp_dtype or 'fp32'}")

    df = load_spam_data(cfg.data_file)
    train_df, val_df, test_df = prepare_splits(
        df,
        balance=cfg.balance,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )

    if cfg.smoke > 0:
        train_df = _limit_split(train_df, cfg.smoke, cfg.seed)
        val_df = _limit_split(val_df, cfg.smoke, cfg.seed)
        test_df = _limit_split(test_df, cfg.smoke, cfg.seed)

    tokenizer = get_tokenizer("gpt2")
    pad_token_id = tokenizer.special_token_id(EOS_TOKEN)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        tokenizer,
        pad_token_id=pad_token_id,
        max_length=cfg.max_length,
        padding=cfg.padding,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    if verbose:
        print(
            f"Datos: {len(df):,} mensajes | "
            f"train {len(train_df):,} ({train_df['label'].value_counts().to_dict()}) | "
            f"val {len(val_df):,} | test {len(test_df):,}"
        )

    model, config = SpamClassifier.from_pretrained(
        cfg.model, device=device, drop_rate=cfg.drop_rate
    )
    n_blocks = configure_trainable(model, cfg.trainable, cfg.unfreeze_last)
    n_trainable = count_trainable_parameters(model)
    n_total = count_total_parameters(model)

    if verbose:
        print(
            f"GPT-2: {resolve_model_name(cfg.model)} | "
            f"emb_dim={config.emb_dim} | bloques={config.layers} | "
            f"descongelados={n_blocks}"
        )
        print(
            f"Parámetros entrenables: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / max(1, n_total):.1f}%)"
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df, device))
    optimizer = torch.optim.AdamW(
        split_param_groups(
            model,
            lr=cfg.lr,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
    )

    head_only_cached = cfg.cache_features and not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )

    train_features = val_features = test_features = None
    train_head_labels = val_head_labels = test_head_labels = None

    feature_start = time.perf_counter()
    if head_only_cached:
        if verbose:
            print("Extrayendo features del backbone (backbone congelado)...")
        train_features, train_head_labels = extract_features(
            model, train_loader, device
        )
        val_features, val_head_labels = extract_features(model, val_loader, device)
        test_features, test_head_labels = extract_features(model, test_loader, device)
        train_features = train_features.to(device)
        val_features = val_features.to(device)
        test_features = test_features.to(device)
        train_head_labels = train_head_labels.to(device)
        val_head_labels = val_head_labels.to(device)
        test_head_labels = test_head_labels.to(device)

        steps_per_epoch = math.ceil(len(train_features) / cfg.batch_size)
    else:
        steps_per_epoch = len(train_loader)

    feature_time = time.perf_counter() - feature_start
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    best_val_f1 = -1.0
    best_threshold = 0.5
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    train_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        if head_only_cached:
            train_loss = train_head_epoch(
                model,
                train_features,
                train_head_labels,
                optimizer,
                scheduler,
                criterion,
                batch_size=cfg.batch_size,
                grad_clip=cfg.grad_clip,
            )
            val_loss, val_probs, val_labels = evaluate_features(
                model, val_features, val_head_labels
            )
        else:
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                amp_dtype,
                cfg.grad_clip,
                cfg.log_every,
                epoch,
                cfg.epochs,
            )
            val_loss, val_probs, val_labels = evaluate_loader(
                model, val_loader, device
            )

        threshold, val_metrics = tune_threshold(val_probs, val_labels)
        val_f1 = val_metrics["f1"]
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_f1,
                "threshold": threshold,
                "lr": lr,
                "elapsed_s": elapsed,
            }
        )

        improved = val_f1 > best_val_f1 + cfg.min_delta
        if improved:
            best_val_f1 = val_f1
            best_threshold = threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if name in {n for n, _ in trainable_parameters(model)}
            }
            save_checkpoint(
                out_dir / "best.pt",
                model,
                config,
                cfg,
                best_threshold,
                val_metrics,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            out_dir / "last.pt",
            model,
            config,
            cfg,
            threshold,
            val_metrics,
        )

        if verbose:
            print(
                f"Época {epoch + 1}/{cfg.epochs} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"val F1 {val_f1:.4f} | thr {threshold:.2f} | "
                f"acc {val_metrics['accuracy']:.4f} | {elapsed:.1f}s"
                + (" | mejor" if improved else "")
            )

        if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
            if verbose:
                print(f"Early stopping: {cfg.patience} épocas sin mejorar.")
            break

    train_time = time.perf_counter() - train_start

    if best_state is None:
        raise RuntimeError("No se guardó ningún checkpoint; revisa el entrenamiento.")

    model.load_state_dict(best_state, strict=False)
    if best_epoch < 0:
        best_epoch = len(history)

    if head_only_cached:
        test_loss, test_probs, test_labels = evaluate_features(
            model, test_features, test_head_labels
        )
    else:
        test_loss, test_probs, test_labels = evaluate_loader(
            model, test_loader, device
        )

    test_metrics = binary_metrics(test_probs, test_labels, best_threshold)
    test_metrics_at_half = binary_metrics(test_probs, test_labels, 0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)

    metrics = {
        "config": asdict(cfg),
        "base_model": resolve_model_name(cfg.model),
        "model_config": config_to_dict(config),
        "trainable_blocks": n_blocks,
        "trainable_parameters": n_trainable,
        "total_parameters": n_total,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_counts": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "threshold": best_threshold,
        "test": test_metrics,
        "test_at_0.5": test_metrics_at_half,
        "feature_time_s": feature_time,
        "train_time_s": train_time,
        "total_time_s": time.perf_counter() - total_start,
        "history": history,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    if verbose:
        print(
            f"Test | acc {test_metrics['accuracy']:.4f} | "
            f"precision {test_metrics['precision']:.4f} | "
            f"recall {test_metrics['recall']:.4f} | F1 {test_metrics['f1']:.4f} | "
            f"thr {best_threshold:.2f}"
        )
        print(f"Checkpoints y métricas en {out_dir}")

    return {
        "run_id": out_dir.name,
        "model": cfg.model,
        "trainable": cfg.trainable,
        "trainable_params": n_trainable,
        "padding": cfg.padding,
        "balance": cfg.balance,
        "max_len": cfg.max_length,
        "epochs": cfg.epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "threshold": best_threshold,
        "train_time_s": train_time + feature_time,
        "feature_time_s": feature_time,
        "total_time_s": metrics["total_time_s"],
        "out_dir": str(out_dir),
    }


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de clasificación de SMS spam con GPT-2"
    )

    parser.add_argument(
        "--model",
        default="small",
        help="small | medium | large | xl (o gpt2, gpt2-medium, ...)",
    )
    parser.add_argument(
        "--trainable",
        default="last_2_blocks",
        choices=("head", "last_block", "last_2_blocks", "last_4_blocks", "full"),
        help="Parte del backbone que se entrena",
    )
    parser.add_argument(
        "--unfreeze-last",
        type=int,
        default=None,
        help="Alternativa genérica: descongela los últimos N bloques",
    )
    parser.add_argument(
        "--padding", default="dynamic", choices=("dynamic", "fixed")
    )
    parser.add_argument(
        "--balance", default="none", choices=("none", "undersample")
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR del backbone")
    parser.add_argument("--head-lr", type=float, default=5e-4, help="LR de la cabeza")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Limita cada split a N ejemplos (0 = sin límite)",
    )
    parser.add_argument(
        "--no-cache-features",
        action="store_false",
        dest="cache_features",
        help="Para --trainable head: no cachear features del backbone",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data", default=str(DATA_FILE_PATH))

    args = parser.parse_args()
    return TrainingConfig(
        model=args.model,
        trainable=args.trainable,
        unfreeze_last=args.unfreeze_last,
        padding=args.padding,
        balance=args.balance,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        drop_rate=args.drop_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        log_every=args.log_every,
        smoke=args.smoke,
        cache_features=args.cache_features,
        out_dir=args.out_dir,
        data_file=args.data,
    )


if __name__ == "__main__":
    run_training(parse_args())
