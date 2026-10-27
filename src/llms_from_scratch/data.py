from pathlib import Path

import torch
from torch.utils.data import Dataset

from llms_from_scratch.tokenizer import Tokenizer


class LLMDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self._build(text, tokenizer, context_length, stride)

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: Tokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> "LLMDataset":
        dataset = cls.__new__(cls)
        dataset._build(text, tokenizer, context_length, stride)
        return dataset

    def _build(
        self,
        text: str,
        tokenizer: Tokenizer,
        context_length: int,
        stride: int | None,
    ) -> None:
        self.context_length = context_length
        stride = stride or context_length

        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.starts = range(0, len(self.tokens) - context_length, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.starts[index]
        x = self.tokens[i : i + self.context_length]
        y = self.tokens[i + 1 : i + self.context_length + 1]
        return x, y
