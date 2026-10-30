#!/usr/bin/env bash
# Prepara el entorno del proyecto.
#
# En Colab reutiliza el torch preinstalado y solo instala lo que falte.
# En local crea el entorno con uv.
#
# Uso:
#   ./scripts/setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_colab.sh
source "$ROOT/scripts/_colab.sh"

# En Colab no creamos un .venv: el runtime ya trae torch compilado para su
# GPU/TPU y reinstalarlo con uv suele tardar y perder esas optimizaciones.
if is_colab; then
    echo "Colab detectado: se usa el torch preinstalado (sin uv sync)."
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

    python - <<'PY'
import importlib.util
import subprocess
import sys

# Solo instalamos lo que no venga ya en Colab; torch y torchvision se respetan.
missing = [
    package
    for package in ("tiktoken", "pandas")
    if importlib.util.find_spec(package) is None
]
if missing:
    print("Instalando:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY

    python - <<'PY'
import pandas
import tiktoken
import torch

print(f"torch {torch.__version__} (preinstalado de Colab)")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"tiktoken {tiktoken.__version__}")
print(f"pandas {pandas.__version__}")
PY

    echo
    echo "Entorno listo."
    echo "Entrenar en GPU/CPU:  bash scripts/train.sh --epochs 1"
    echo "Entrenar en TPU:      bash scripts/train_tpu.sh --epochs 1"
    echo "Generar texto:        bash scripts/generate.sh \"En un lugar de la Mancha\""
    exit 0
fi

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
