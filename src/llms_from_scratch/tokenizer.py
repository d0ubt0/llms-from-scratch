# %%
from typing import Literal

import tiktoken
import torch
from tiktoken import Encoding, get_encoding
from torch import Tensor

# %%


class Tokenizer:
    def __init__(self, tokenizer: Encoding) -> None:
        self.tokenizer = tokenizer

    def decode(self, tokens: Tensor) -> str:
        flat = tokens.flatten()
        out = self.tokenizer.decode(flat.tolist())
        return out

    def encode(self, string: str) -> Tensor:
        out = self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})
        out = torch.tensor(out).unsqueeze(0)
        return out


# %%
def get_tokenizer(name_tokenizer: Literal["gpt2"]) -> Tokenizer:
    t = tiktoken.get_encoding(name_tokenizer)
    return Tokenizer(t)
