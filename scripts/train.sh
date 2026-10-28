#!/usr/bin/env bash
# Ejecuta el entrenamiento del LLM.
#
# Uso:
#   ./scripts/train.sh
#   ./scripts/train.sh --epochs 3 --batch-size 8
#
# Variables de entorno (opcionales):
#   DATA        corpus de entrenamiento   (data/don_quijote.txt)
#   CONFIG      micro | small             (micro)
#   DEVICE      auto | cpu | cuda | xla   (auto)
#   EPOCHS      número de épocas          (1)
#   BATCH_SIZE  tamaño de batch           (4)
#   OUT_DIR     carpeta de checkpoints    (checkpoints)
#   USE_UV      1 usa uv, 0 usa python    (1; en Colab siempre se usa python)
#
# Ejemplos:
#   DATA=data/mi_corpus.txt ./scripts/train.sh
#   DEVICE=cuda ./scripts/train.sh --epochs 5
#   USE_UV=0 ./scripts/train.sh   # en Colab sin uv (usa PYTHONPATH=src)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/don_quijote.txt}"
CONFIG="${CONFIG:-micro}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-checkpoints}"

# En Colab usamos siempre el Python del sistema: ya trae el torch compilado
# para su GPU/TPU. Si dejáramos que uv cree un .venv, descargaría otro torch.
if [[ -n "${COLAB_RELEASE_TAG:-}${COLAB_GPU:-}${COLAB_TPU_ADDR:-}" ]]; then
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    RUNNER=(python)
elif [[ "${USE_UV:-1}" == "1" ]] && command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python)
else
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        RUNNER=("$ROOT/.venv/bin/python")
    else
        RUNNER=(python)
    fi
fi

# Los valores efectivos (incluidos los flags pasados en "$@") los imprime
# train.py después de parsear los argumentos.
exec "${RUNNER[@]}" -m llms_from_scratch.train \
    --data "$DATA" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --out-dir "$OUT_DIR" \
    "$@"
