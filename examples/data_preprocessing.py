import os
import re
import urllib.request

import tiktoken

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

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["$unk$", "$end$"])
vocab_size = len(all_tokens)

vocab = {token: i for i, token in enumerate(all_tokens)}


class TokenizerV1:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: t for t, i in vocab.items()}

    def preproccesor(self, raw_text) -> list[str]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]

        preprocessed = [
            token if token in self.str_to_int else "$unk$" for token in preprocessed
        ]
        preprocessed.append("$end$")

        return preprocessed

    def encode(self, text) -> list[int]:
        preprocessed = self.preproccesor(text)

        ids = [self.str_to_int[token] for token in preprocessed]

        return ids

    def decode(self, ids) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        return text


### Bytepair encoding
print("\n\n\n\n")
tokenizer = tiktoken.get_encoding("gpt2")

encoded_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print(len(encoded_text))

encoded_sample = encoded_text[50:]

context_size = 10
x = encoded_sample[:context_size]
y = encoded_sample[1 : context_size + 1]

for i in range(1, context_size + 1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    print(tokenizer.decode(context), "->", tokenizer.decode([desired]))
