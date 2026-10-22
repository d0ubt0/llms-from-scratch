import torch
from torch.utils.data import Dataset

from llms_from_scratch.tokenizer import Tokenizer


class LLMDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        self.context_length = context_length
        stride = stride or context_length

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = tokenizer.encode(text).flatten()
        self.starts = range(0, len(self.tokens) - context_length, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.starts[index]
        x = self.tokens[i : i + self.context_length]
        y = self.tokens[i + 1 : i + self.context_length + 1]
        return x, y
