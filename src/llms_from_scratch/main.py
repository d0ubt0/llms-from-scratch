# %%
import torch

from llms_from_scratch.config import get_config
from llms_from_scratch.llm import LLM
from llms_from_scratch.tokenizer import Tokenizer, get_tokenizer


def main():
    config = get_config("micro")

    tokenizer = get_tokenizer("gpt2")

    inputs = torch.randint(0, 100, (1, 10))

    llm = LLM(config)

    print(llm(inputs))


# %%
if __name__ == "__main__":
    main()
