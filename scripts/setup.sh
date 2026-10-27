#!/usr/bin/env bash
# Prepara el entorno del proyecto con uv.
#
# Uso:
#   ./scripts/setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv no está instalado." >&2
    echo "Instálalo con:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "Sincronizando dependencias con uv..."
uv sync

uv run python - <<'PY'
import pandas
import tiktoken
import torch

print(f"torch {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"tiktoken {tiktoken.__version__}")
print(f"pandas {pandas.__version__}")
PY

echo
echo "Entorno listo."
echo "Entrenar en CPU/CUDA:  ./scripts/train.sh --epochs 1"
echo "Entrenar en TPU:       ./scripts/train_tpu.sh --epochs 1"
echo "Generar texto:         ./scripts/generate.sh \"En un lugar de la Mancha\""
