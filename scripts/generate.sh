#!/usr/bin/env bash
# Genera texto con un checkpoint entrenado e imprime el resultado por consola.
#
# Uso:
#   ./scripts/generate.sh "En un lugar de la Mancha"
#   ./scripts/generate.sh "Había una vez" --max-new-tokens 100 --temperature 0.9
#   ./scripts/generate.sh "Hello, my name is" --pretrained gpt2
#   ./scripts/generate.sh "Hola" --greedy
#
# Variables de entorno (opcionales):
#   CHECKPOINT       ruta al .pt        (checkpoints/best.pt o last.pt)
#   TOKENIZER        tokenizer.json     (el guardado en el checkpoint)
#   DEVICE           auto | cpu | cuda | xla   (auto)
#   MAX_NEW_TOKENS   tokens a generar   (200)
#   TEMPERATURE      temperatura        (0.8)
#   TOP_K            top-k              (40)
#   SEED             semilla            (vacío = aleatoria)
#   USE_UV           1 usa uv, 0 usa python   (1; en Colab siempre python)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/_colab.sh
source "$ROOT/scripts/_colab.sh"

# En Colab usamos siempre el Python del sistema (torch ya viene optimizado).
if is_colab; then
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

if [[ $# -eq 0 ]]; then
    echo "Uso: ./scripts/generate.sh \"tu prompt\" [opciones]" >&2
    echo "Opciones: --checkpoint --pretrained --tokenizer --device" >&2
    echo "          --max-new-tokens --temperature --top-k --seed --greedy" >&2
    exit 1
fi

PROMPT="$1"
shift

if [[ "$PROMPT" == "-h" || "$PROMPT" == "--help" ]]; then
    exec "${RUNNER[@]}" -m llms_from_scratch.generate --help
fi

CHECKPOINT="${CHECKPOINT:-}"
TOKENIZER="${TOKENIZER:-}"
DEVICE="${DEVICE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-}"

if [[ -n "$CHECKPOINT" ]]; then
    set -- --checkpoint "$CHECKPOINT" "$@"
fi

if [[ -n "$TOKENIZER" ]]; then
    set -- --tokenizer "$TOKENIZER" "$@"
fi

if [[ -n "$SEED" ]]; then
    set -- --seed "$SEED" "$@"
fi

exec "${RUNNER[@]}" -m llms_from_scratch.generate \
    "$PROMPT" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    "$@"
