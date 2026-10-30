"""Entrena un tokenizer BPE byte-level (estilo GPT-2) sobre un corpus .txt.

Usa la librería `tokenizers` de Hugging Face. Al ser byte-level, el
vocabulario cubre los 256 valores de byte posibles: cualquier texto se
puede codificar sin token <unk>.

Uso:
    uv run python scripts/train_tokenizer.py
    uv run python scripts/train_tokenizer.py \
        --data data/corpus_es.txt --vocab-size 32_000 \
        --out checkpoints/tokenizer

Genera:
    <out>/tokenizer.json   tokenizer completo (vocab + merges + config)
    <out>/tokenizer_info.json  resumen de la configuración usada

Para usarlo después:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("checkpoints/tokenizer/tokenizer.json")
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
# Tamaño de bloque por defecto al leer el corpus (en millones de caracteres).
DEFAULT_CHUNK_CHARS = 8_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un tokenizer BPE byte-level sobre un corpus .txt"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/don_quijote.txt"),
        help="Corpus de entrada (UTF-8)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/tokenizer"),
        help="Carpeta donde guardar tokenizer.json",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help="Tamaño final del vocabulario, tokens especiales incluidos",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Frecuencia mínima de un par para fusionarse",
    )
    parser.add_argument(
        "--special-tokens",
        default=",".join(DEFAULT_SPECIAL_TOKENS),
        help="Tokens especiales separados por comas (vacío = ninguno)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help="Caracteres leídos por bloque (streaming, no carga todo en RAM)",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=200_000,
        help="Caracteres usados para calcular estadísticas al final",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Longitud máxima al codificar (0 = sin truncar)",
    )
    parser.add_argument(
        "--prefix-space",
        dest="add_prefix_space",
        action="store_true",
        help=(
            "Añadir un espacio inicial al codificar cada texto (como el "
            "tokenizer GPT-2 de HF). Por defecto está desactivado para que "
            "encode/decode sea reversible sobre documentos completos."
        ),
    )
    parser.set_defaults(add_prefix_space=False)
    return parser.parse_args()


def iter_corpus(path: Path, chunk_chars: int) -> Iterator[str]:
    """Lee el corpus en bloques alineados a saltos de línea."""
    chunk_chars = max(1, chunk_chars)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        buffer = ""
        while True:
            block = handle.read(chunk_chars)
            if not block:
                break

            buffer += block
            cut = buffer.rfind("\n")
            if cut >= 0:
                yield buffer[: cut + 1]
                buffer = buffer[cut + 1 :]

        if buffer:
            yield buffer


def build_trainer(args: argparse.Namespace) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens_list,
        # Garantiza que los 256 bytes estén en el vocabulario desde el inicio.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )


def build_tokenizer(args: argparse.Namespace) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=args.add_prefix_space,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()

    if args.max_length > 0:
        tokenizer.enable_truncation(max_length=args.max_length)

    return tokenizer


def train(args: argparse.Namespace) -> Tokenizer:
    if not args.data.exists():
        raise SystemExit(
            f"No existe {args.data}. Genera el corpus antes, por ejemplo con "
            "`uv run python scripts/download_corpus.py`."
        )

    if args.data.stat().st_size == 0:
        raise SystemExit(f"{args.data} está vacío.")

    if args.vocab_size <= len(args.special_tokens_list) + 256:
        raise SystemExit(
            "--vocab-size debe ser mayor que 256 + tokens especiales "
            f"(recibido {args.vocab_size})."
        )

    size_mb = args.data.stat().st_size / 1_000_000
    print(f"Corpus: {args.data} ({size_mb:.2f} MB)")
    print(
        f"Vocabulario objetivo: {args.vocab_size:,} | "
        f"min_frequency: {args.min_frequency} | "
        f"especiales: {args.special_tokens_list or 'ninguno'}"
    )

    tokenizer = build_tokenizer(args)
    started = time.perf_counter()

    tokenizer.train_from_iterator(
        iter_corpus(args.data, args.chunk_chars),
        trainer=build_trainer(args),
        length=None,
    )

    elapsed = time.perf_counter() - started
    print(f"Entrenamiento terminado en {elapsed:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer guardado en: {tokenizer_path}")

    info = {
        "data": str(args.data),
        "corpus_bytes": args.data.stat().st_size,
        "vocab_size": tokenizer.get_vocab_size(),
        "vocab_size_requested": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": args.special_tokens_list,
        "add_prefix_space": args.add_prefix_space,
    }
    (args.out / "tokenizer_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return tokenizer


def report(tokenizer: Tokenizer, args: argparse.Namespace) -> None:
    if args.sample_chars <= 0:
        return

    with args.data.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(args.sample_chars)

    if not sample:
        return

    started = time.perf_counter()
    encoding = tokenizer.encode(sample)
    elapsed = time.perf_counter() - started

    num_tokens = len(encoding.ids)
    sample_bytes = len(sample.encode("utf-8"))
    tokens_per_byte = num_tokens / max(1, sample_bytes)
    total_bytes = args.data.stat().st_size
    estimated = int(total_bytes * tokens_per_byte)

    print()
    print(f"Vocabulario final: {tokenizer.get_vocab_size():,} tokens")
    print(
        f"Muestra: {len(sample):,} caracteres -> {num_tokens:,} tokens | "
        f"{len(sample) / max(1, num_tokens):.2f} caracteres/token"
    )
    print(
        f"Tokens estimados en el corpus: ~{estimated:,} "
        f"({elapsed:.2f}s codificar la muestra)"
    )

    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    # Con --prefix-space el tokenizer antepone un espacio al texto codificado.
    expected = " " + sample if args.add_prefix_space else sample
    state = "OK" if decoded == expected else "FALLA"
    print(f"Round-trip encode/decode: {state}")

    if args.special_tokens_list:
        for token in args.special_tokens_list:
            probe = tokenizer.encode(f"hola {token} mundo")
            state = "OK" if token in probe.tokens else "FALLA"
            print(f"Token especial {token!r}: {state}")

    snippet = encoding.tokens[:60]
    print(f"Primeras piezas: {'|'.join(snippet)}")


def main() -> None:
    args = parse_args()
    args.special_tokens_list = [
        token.strip() for token in args.special_tokens.split(",") if token.strip()
    ]

    tokenizer = train(args)
    report(tokenizer, args)


if __name__ == "__main__":
    main()
