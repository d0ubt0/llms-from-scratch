import os
import re
import urllib.request
from typing import Any

import tiktoken
import torch
from sympy import false
from torch import tensor
from torch.utils.data import DataLoader, Dataset

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

FILE_PATH = "data/the-verdict.txt"

os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)

urllib.request.urlretrieve(url, FILE_PATH)

# with open(FILE_PATH, "r", encoding="utf-8") as file:
#     raw_text = file.read()

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

# preprocessed = [word.strip() for word in preprocessed if word.strip()]

# all_tokens = sorted(set(preprocessed))
# all_tokens.extend(["$unk$", "$end$"])
# vocab_size = len(all_tokens)

# vocab = {token: i for i, token in enumerate(all_tokens)}


# class TokenizerV1:
#     def __init__(self, vocab) -> None:
#         self.str_to_int = vocab
#         self.int_to_str = {i: t for t, i in vocab.items()}

#     def preproccesor(self, raw_text) -> list[str]:
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
#         preprocessed = [word.strip() for word in preprocessed if word.strip()]

#         preprocessed = [
#             token if token in self.str_to_int else "$unk$" for token in preprocessed
#         ]
#         preprocessed.append("$end$")

#         return preprocessed

#     def encode(self, text) -> list[int]:
#         preprocessed = self.preproccesor(text)

#         ids = [self.str_to_int[token] for token in preprocessed]

#         return ids

#     def decode(self, ids) -> str:
#         text = " ".join([self.int_to_str[id] for id in ids])
#         return text


### Bytepair encoding
# print("\n\n\n\n")
# tokenizer = tiktoken.get_encoding("gpt2")

# encoded_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
# print(len(encoded_text))

# encoded_sample = encoded_text[50:]

# context_size = 10
# x = encoded_sample[:context_size]
# y = encoded_sample[1 : context_size + 1]

# for i in range(1, context_size + 1):
#     context = encoded_sample[:i]
#     desired = encoded_sample[i]
#     print(tokenizer.decode(context), "->", tokenizer.decode([desired]))


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_window_length, window_stride) -> None:
        self.input_tokens = []
        self.output_tokens = []

        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - max_window_length, window_stride):
            input_chunk = tokens[i : i + max_window_length]
            output_chunk = tokens[i + 1 : i + max_window_length + 1]

            self.input_tokens.append(tensor(input_chunk))
            self.output_tokens.append(tensor(output_chunk))

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, index):
        return self.input_tokens[index], self.output_tokens[index]


def create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.encoding_for_model("gpt2")

    dataset = GPTDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, batch_size, shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader


# with open(FILE_PATH, "r", encoding="utf-8") as file:
#     raw_text = file.read()

# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=2, stride=2, shuffle=False
# )

# dataloader = iter(dataloader)
# print(next(dataloader))
# print(next(dataloader))

## How embedding works, a simple example
input_ids = tensor([1, 5, 3, 4])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(input_ids))
