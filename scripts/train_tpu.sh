#!/usr/bin/env bash
# Entrena en TPU (Colab o TPU VM) usando PyTorch/XLA.
#
# Requisitos:
#   pip install "torch_xla==<versión-de-torch>"   # deben coincidir
#
# Uso:
#   ./scripts/train_tpu.sh
#   ./scripts/train_tpu.sh --epochs 3 --batch-size 8
#
# En Colab no hace falta USE_UV: los scripts usan el Python del sistema.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# En Colab, comprobamos torch_xla antes de arrancar para dar un error claro.
# Colab ya trae torch: instalar torch_xla sin fijar versión puede reemplazarlo.
if [[ -n "${COLAB_RELEASE_TAG:-}${COLAB_GPU:-}${COLAB_TPU_ADDR:-}" ]]; then
    if ! python -c "import torch_xla" >/dev/null 2>&1; then
        TORCH_VERSION="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
        echo "Error: falta torch_xla (o no coincide con torch)." >&2
        echo "Instálalo con la MISMA versión de torch:" >&2
        echo "  %pip install \"torch_xla==${TORCH_VERSION}\"" >&2
        exit 1
    fi
fi

# torch_xla debe saber el backend antes de importarse.
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export DEVICE="${DEVICE:-xla}"

exec "$ROOT/scripts/train.sh" "$@"
