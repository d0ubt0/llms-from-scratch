#!/usr/bin/env bash
# Utilidad compartida: detecta si se ejecuta en Google Colab.
#
# Se usa con `source` desde los demás scripts:
#   source "$ROOT/scripts/_colab.sh"
#   if is_colab; then ...; fi
#
# La terminal de Colab NO hereda las variables COLAB_* del notebook, y
# COLAB_GPU puede existir pero estar vacía, así que no basta con mirar el
# valor de las variables. Se comprueba, en orden:
#   1. LLMS_COLAB=1|0  fuerza el resultado (útil para depurar).
#   2. Variables COLAB_* definidas (aunque vacías).
#   3. Existe /content y el paquete google.colab es importable.

is_colab() {
    case "${LLMS_COLAB:-}" in
        1) return 0 ;;
        0) return 1 ;;
    esac

    if [[ -n "${COLAB_RELEASE_TAG+x}${COLAB_GPU+x}${COLAB_TPU_ADDR+x}" ]]; then
        return 0
    fi

    if [[ -d /content ]] && python3 -c "import google.colab" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}
