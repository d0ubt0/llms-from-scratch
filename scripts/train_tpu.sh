#!/usr/bin/env bash
# Entrena en TPU (Colab o TPU VM) usando PyTorch/XLA.
#
# Requisitos:
#   pip install torch_xla[tpu]     # versión igual a la de torch
#
# Uso:
#   ./scripts/train_tpu.sh
#   ./scripts/train_tpu.sh --epochs 3 --batch-size 8
#   USE_UV=0 ./scripts/train_tpu.sh   # en Colab sin uv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# torch_xla debe saber el backend antes de importarse.
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export DEVICE="${DEVICE:-xla}"

exec "$ROOT/scripts/train.sh" "$@"
