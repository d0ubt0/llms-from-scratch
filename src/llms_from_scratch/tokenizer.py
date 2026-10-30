# %%
from pathlib import Path
from typing import Literal, Protocol

import tiktoken
from tiktoken import Encoding
from tokenizers import Tokenizer as HFTokenizerBackend

# %%
EOS_TOKEN = "<|endoftext|>"


class TextTokenizer(Protocol):
    """Interfaz mínima que consumen `LLMDataset` y `generate`."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    def special_token_id(self, token: str) -> int | None: ...


class Tokenizer:
    def __init__(self, tokenizer: Encoding) -> None:
        self.tokenizer = tokenizer

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, allowed_special={EOS_TOKEN})

    def special_token_id(self, token: str) -> int | None:
        try:
            return self.tokenizer.encode_single_token(token)
        except KeyError:
            return None


class HFTokenizer:
    """Adaptador para un BPE byte-level entrenado con `scripts/train_tokenizer.py`.

    Se desactivan el truncado y el padding para que codificar un documento
    completo nunca recorte tokens en silencio.
    """

    def __init__(self, tokenizer: HFTokenizerBackend) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "HFTokenizer":
        return cls(HFTokenizerBackend.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        # Los tokens especiales (p. ej. <|endoftext|>) forman parte del texto.
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def special_token_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)


# %%
def get_tokenizer(name_tokenizer: Literal["gpt2"]) -> Tokenizer:
    t = tiktoken.get_encoding(name_tokenizer)
    return Tokenizer(t)


def get_hf_tokenizer(path: str | Path) -> HFTokenizer:
    """Carga un tokenizer.json creado con `scripts/train_tokenizer.py`."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No existe el tokenizer {path}. Créalo con "
            "`uv run python scripts/train_tokenizer.py`."
        )
    return HFTokenizer.from_file(path)
