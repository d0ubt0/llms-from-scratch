#!/usr/bin/env python3
"""Genera la figura del barrido de fine-tuning con seaborn.

Lee ``checkpoints/experiments/results.csv`` y escribe
``docs/figures/f1_por_scope.png``. Requiere pandas, matplotlib y seaborn.

Ejemplo::

    python scripts/plot_experiments.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCOPE_LABELS = {
    "head": "Solo capa final",
    "last_block": "Último bloque",
    "last_2_blocks": "Últimos 2 bloques",
    "full": "Modelo completo",
}
SCOPE_ORDER = list(SCOPE_LABELS.values())
PADDING_LABELS = {"dynamic": "dinámica", "fixed": "fija"}
PADDING_COLORS = {"dinámica": "#2a9d8f", "fija": "#e9c46a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("checkpoints/experiments/results.csv"),
        help="CSV agregado del barrido",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Carpeta donde guardar la figura",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)

    df = pd.read_csv(args.results)
    df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df = df[df["balance"] == "none"].copy()
    df["scope"] = df["trainable"].map(SCOPE_LABELS)
    df["preparacion"] = df["padding"].map(PADDING_LABELS)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df,
        x="scope",
        y="test_f1",
        hue="preparacion",
        order=SCOPE_ORDER,
        hue_order=["dinámica", "fija"],
        palette=PADDING_COLORS,
        errorbar=None,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.3f}", padding=2, fontsize=8)

    ax.set_ylim(0.88, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Resultado (F1)")
    ax.set_title("Resultado según qué parte del modelo entrenamos")
    ax.legend(title="Preparación", loc="upper left")
    sns.despine(fig)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "f1_por_scope.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
