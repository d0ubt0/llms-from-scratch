"""Barrido de configuraciones de fine-tuning para SMS spam.

Compara padding dinámico/fijo, distintas partes entrenables (linear probe,
últimos bloques, fine-tuning completo), balanceo de clases y tamaños de GPT-2.
Cada corrida usa el mismo split y semilla, y escribe sus métricas en
``<out-dir>/<run_id>/metrics.json``. Al final se agregan todas en
``results.csv`` y ``results.md``.

Ejemplos::

    uv run python -m llms_from_scratch.finetuning.experiments --preset quick
    uv run python -m llms_from_scratch.finetuning.experiments --preset full
    uv run python -m llms_from_scratch.finetuning.experiments \
        --models small medium --scopes head last_2_blocks \
        --paddings dynamic fixed --balances none
"""

from __future__ import annotations

import argparse
import traceback
from itertools import product
from pathlib import Path

import pandas as pd

from llms_from_scratch.finetuning.data import DATA_FILE_PATH
from llms_from_scratch.finetuning.model import (
    TRAINABLE_SCOPES,
    resolve_model_name,
)
from llms_from_scratch.finetuning.train import TrainingConfig, run_training

PRESETS = {
    "smoke": {
        "models": ["small"],
        "paddings": ["dynamic"],
        "scopes": ["last_2_blocks"],
        "balances": ["none"],
        "epochs": 1,
        "patience": 0,
        "smoke": 200,
    },
    "quick": {
        "models": ["small"],
        "paddings": ["dynamic"],
        "scopes": ["head", "last_2_blocks"],
        "balances": ["none"],
        "epochs": 3,
        "patience": 2,
        "smoke": 0,
    },
    "full": {
        "models": ["small"],
        "paddings": ["dynamic", "fixed"],
        "scopes": ["head", "last_block", "last_2_blocks", "full"],
        "balances": ["none", "undersample"],
        "epochs": 3,
        "patience": 2,
        "smoke": 0,
    },
    "custom": {
        "models": ["small"],
        "paddings": ["dynamic"],
        "scopes": ["last_2_blocks"],
        "balances": ["none"],
        "epochs": 3,
        "patience": 2,
        "smoke": 0,
    },
}

RESULT_COLUMNS = [
    "run_id",
    "model",
    "trainable",
    "trainable_params",
    "padding",
    "balance",
    "max_len",
    "epochs",
    "best_epoch",
    "best_val_f1",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1",
    "threshold",
    "train_time_s",
    "feature_time_s",
    "total_time_s",
    "out_dir",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara configuraciones de fine-tuning de SMS spam"
    )
    parser.add_argument(
        "--preset",
        default="quick",
        choices=tuple(PRESETS),
        help="smoke = prueba mínima; quick = comparación corta; full = matriz completa",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--paddings",
        nargs="+",
        default=None,
        choices=("dynamic", "fixed"),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=None,
        choices=TRAINABLE_SCOPES,
    )
    parser.add_argument(
        "--balances",
        nargs="+",
        default=None,
        choices=("none", "undersample"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--smoke", type=int, default=None)

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cachea features del backbone cuando --trainable head",
    )
    parser.add_argument("--out-dir", default="checkpoints/experiments")
    parser.add_argument("--data", default=str(DATA_FILE_PATH))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_combinations(args: argparse.Namespace) -> list[dict]:
    preset = PRESETS[args.preset]

    models = args.models if args.models is not None else preset["models"]
    paddings = args.paddings if args.paddings is not None else preset["paddings"]
    scopes = args.scopes if args.scopes is not None else preset["scopes"]
    balances = args.balances if args.balances is not None else preset["balances"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    patience = args.patience if args.patience is not None else preset["patience"]
    smoke = args.smoke if args.smoke is not None else preset["smoke"]

    # "small" y "gpt2" son el mismo modelo: evita corridas duplicadas.
    unique_models: list[tuple[str, str]] = []
    seen_models: set[str] = set()
    for model in models:
        resolved = resolve_model_name(model)
        if resolved not in seen_models:
            seen_models.add(resolved)
            unique_models.append((model, resolved))

    combinations = []
    for (model, resolved), padding, scope, balance in product(
        unique_models, paddings, scopes, balances
    ):
        combinations.append(
            {
                "model": model,
                "resolved_model": resolved,
                "padding": padding,
                "scope": scope,
                "balance": balance,
                "epochs": epochs,
                "patience": patience,
                "smoke": smoke,
            }
        )
    return combinations


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [column for column in RESULT_COLUMNS if column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]

    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif column in {"train_time_s", "feature_time_s", "total_time_s"}:
                cells.append(f"{float(value):.1f}")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            elif column == "trainable_params":
                cells.append(f"{int(value):,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    combinations = build_combinations(args)

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Corridas: {len(combinations)} | resultados en {root}", flush=True)
    rows: list[dict] = []

    for index, combination in enumerate(combinations, start=1):
        run_id = (
            f"{combination['resolved_model']}__{combination['scope']}__"
            f"{combination['padding']}__{combination['balance']}__"
            f"len{args.max_length}"
        )
        run_dir = root / run_id

        print(
            f"\n[{index}/{len(combinations)}] {run_id} "
            f"({combination['epochs']} épocas)",
            flush=True,
        )

        config = TrainingConfig(
            model=combination["model"],
            trainable=combination["scope"],
            padding=combination["padding"],
            balance=combination["balance"],
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=combination["epochs"],
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
            patience=combination["patience"],
            log_every=args.log_every,
            smoke=combination["smoke"],
            cache_features=args.cache_features,
            out_dir=str(run_dir),
            data_file=args.data,
        )

        try:
            row = run_training(config, verbose=not args.quiet)
        except Exception as error:  # noqa: BLE001 - una corrida no debe parar el barrido
            traceback.print_exc()
            row = {
                "run_id": run_id,
                "model": combination["model"],
                "trainable": combination["scope"],
                "padding": combination["padding"],
                "balance": combination["balance"],
                "max_len": args.max_length,
                "epochs": combination["epochs"],
                "out_dir": str(run_dir),
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(row)

    results = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = None
    results = results[RESULT_COLUMNS]

    if "test_f1" in results.columns and results["test_f1"].notna().any():
        results = results.sort_values(
            "test_f1", ascending=False, na_position="last"
        ).reset_index(drop=True)

    results.to_csv(root / "results.csv", index=False)
    markdown = dataframe_to_markdown(results)
    (root / "results.md").write_text(
        "# Resultados de fine-tuning SMS spam\n\n" + markdown + "\n",
        encoding="utf-8",
    )

    print("\n" + markdown)
    print(f"\nResultados guardados en {root / 'results.csv'} y {root / 'results.md'}")


if __name__ == "__main__":
    main()
