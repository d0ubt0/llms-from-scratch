import os
import re
import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

FILE_PATH = "data/the-verdict.txt"

os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

urllib.request.urlretrieve(url, FILE_PATH)

with open(FILE_PATH, "r", encoding="utf-8") as file:
    raw_text = file.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

preprocessed = [word.strip() for word in preprocessed if word.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token: i for i, token in enumerate(all_words)}


class TokenizerV1:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: t for t, i in vocab.items()}

    def preproccesor(self, raw_text) -> list[str]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]

        return preprocessed

    def encode(self, text) -> list[int]:
        preprocessed = self.preproccesor(text)

        ids = [self.str_to_int[token] for token in preprocessed]

        return ids

    def decode(self, ids) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        return text


tokenizer = TokenizerV1(vocab)

text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)
