# %%
from typing import Literal

import tiktoken
from tiktoken import Encoding, get_encoding


# %%
class Tokenizer:
    def __init__(self, tokenizer: Encoding) -> None:
        self.tokenizer = tokenizer

    def decode(self, tokens: list[int]) -> str:
        out = self.tokenizer.decode(tokens)
        return out

    def encode(self, string: str) -> list[int]:
        out = self.tokenizer.encode(string)
        return out


# %%
def get_tokenizer(name_tokenizer: Literal["gpt2"]) -> Tokenizer:
    t = tiktoken.get_encoding(name_tokenizer)
    return Tokenizer(t)


# %%

if __name__ == "__main__":
    encoder = get_encoding("gpt2")
    test = "aura chicos aura"

    tokenizer = Tokenizer(encoder)

    tokens = tokenizer.encode(test)
    print("tokens:", tokens)
    decoded = tokenizer.decode(tokens)
    print("decode tokens:", decoded)
