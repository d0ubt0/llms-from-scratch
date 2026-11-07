#!/usr/bin/env bash
# Fine-tuning de clasificación de SMS spam.
#
# Uso:
#   ./scripts/finetune_spam.sh
#   ./scripts/finetune_spam.sh --model medium --trainable head
#   ./scripts/finetune_spam.sh --padding fixed --balance undersample
#
# Variables de entorno (opcionales):
#   MODEL       small | medium | large | xl   (small)
#   TRAINABLE   head | last_block | last_2_blocks | last_4_blocks | full
#                                             (last_2_blocks)
#   PADDING     dynamic | fixed               (dynamic)
#   BALANCE     none | undersample            (none)
#   EPOCHS      número de épocas              (3)
#   BATCH_SIZE  tamaño de batch               (8)
#   MAX_LENGTH  tokens por mensaje            (128)
#   OUT_DIR     carpeta de checkpoints        (checkpoints/spam)
#   DEVICE      auto | cpu | cuda             (auto)
#   USE_UV      1 usa uv, 0 usa python        (1; en Colab siempre se usa python)
#   PYTHON_BIN  intérprete alternativo        (python)
#   LLMS_COLAB  1 fuerza modo Colab por SSH: usa el python del sistema sin .venv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_colab.sh
source "$ROOT/scripts/_colab.sh"

MODEL="${MODEL:-small}"
TRAINABLE="${TRAINABLE:-last_2_blocks}"
PADDING="${PADDING:-dynamic}"
BALANCE="${BALANCE:-none}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OUT_DIR="${OUT_DIR:-checkpoints/spam}"
DEVICE="${DEVICE:-auto}"

if is_colab; then
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    RUNNER=("${PYTHON_BIN:-python}")
elif [[ "${USE_UV:-1}" == "1" ]] && command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run python)
else
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        RUNNER=("$ROOT/.venv/bin/python")
    else
        RUNNER=("${PYTHON_BIN:-python}")
    fi
fi

exec "${RUNNER[@]}" -m llms_from_scratch.finetuning.train \
    --model "$MODEL" \
    --trainable "$TRAINABLE" \
    --padding "$PADDING" \
    --balance "$BALANCE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    "$@"
