# %%
from typing import Literal

import tiktoken
from tiktoken import Encoding

# %%


class Tokenizer:
    def __init__(self, tokenizer: Encoding) -> None:
        self.tokenizer = tokenizer

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})


# %%
def get_tokenizer(name_tokenizer: Literal["gpt2"]) -> Tokenizer:
    t = tiktoken.get_encoding(name_tokenizer)
    return Tokenizer(t)
