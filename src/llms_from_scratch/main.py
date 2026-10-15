# %%
from llms_from_scratch.config import get_config
from llms_from_scratch.tokenizer import Tokenizer, get_tokenizer


def main():
    config = get_config("micro")

    tokenizer = get_tokenizer("gpt2")


# %%
if __name__ == "__main__":
    main()
