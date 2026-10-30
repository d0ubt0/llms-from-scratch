from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset

from llms_from_scratch.tokenizer import TextTokenizer

# Caracteres por bloque al tokenizar. Codificar un corpus grande de una sola
# vez hace que `tokenizers` construya un `Encoding` enorme (ids, tokens,
# offsets y máscaras de cada token) y el proceso muere por falta de RAM.
DEFAULT_CHUNK_CHARS = 1_000_000


def iter_chunks(
    text: str,
    start: int = 0,
    end: int | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> Iterator[str]:
    """Trocea ``text[start:end]`` en bloques sin partir palabras.

    El corte se hace al final de una línea y sin dejar espacios en blanco al
    final del bloque: si un bloque terminara en un salto de línea, ese bloque
    de espacios se tokenizaría distinto que en el texto completo (el
    pre-tokenizer de GPT-2 agrupa los espacios según lo que viene después).
    """
    end = len(text) if end is None else min(end, len(text))
    chunk_chars = max(1, chunk_chars)

    position = start
    while position < end:
        limit = min(position + chunk_chars, end)

        if limit < end:
            cut = text.rfind("\n", position, limit)
            if cut > position:
                # Retrocede sobre los espacios/tabs al final de la línea.
                while cut > position and text[cut - 1].isspace():
                    cut -= 1
                if cut > position:
                    limit = cut

        yield text[position:limit]
        position = limit


def encode_chunks(
    tokenizer: TextTokenizer,
    chunks: Iterable[str],
) -> torch.Tensor:
    """Codifica los bloques y los concatena sin listas gigantes en memoria."""
    pieces: list[torch.Tensor] = []

    for chunk in chunks:
        ids = tokenizer.encode(chunk)
        if ids:
            pieces.append(torch.tensor(ids, dtype=torch.long))

    if not pieces:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(pieces)


class LLMDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: TextTokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self._build(text, 0, len(text), tokenizer, context_length, stride)

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: TextTokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> "LLMDataset":
        return cls.from_range(text, 0, len(text), tokenizer, context_length, stride)

    @classmethod
    def from_range(
        cls,
        text: str,
        start: int,
        end: int,
        tokenizer: TextTokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> "LLMDataset":
        """Usa solo ``text[start:end]`` sin copiar el texto completo."""
        dataset = cls.__new__(cls)
        dataset._build(text, start, end, tokenizer, context_length, stride)
        return dataset

    def _build(
        self,
        text: str,
        start: int,
        end: int,
        tokenizer: TextTokenizer,
        context_length: int,
        stride: int | None,
    ) -> None:
        self.context_length = context_length
        stride = stride or context_length

        self.tokens = encode_chunks(tokenizer, iter_chunks(text, start, end))
        self.starts = range(0, len(self.tokens) - context_length, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.starts[index]
        x = self.tokens[i : i + self.context_length]
        y = self.tokens[i + 1 : i + self.context_length + 1]
        return x, y
