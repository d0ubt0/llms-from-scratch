# %%
import torch

from llms_from_scratch.config import get_config
from llms_from_scratch.llm import LLM
from llms_from_scratch.tokenizer import get_tokenizer


def generate_tokens(model, tokens: torch.Tensor, context_size, max_length=3):
    for _ in range(max_length):
        with torch.no_grad():
            output = model(tokens[:, -context_size:])
        output = output[:, -1, :]
        next_token = torch.argmax(output, dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def main():
    config = get_config("micro")

    tokenizer = get_tokenizer("gpt2")

    text = "How are you"
    inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    llm = LLM(config)

    generated = generate_tokens(llm, inputs, context_size=15)
    print(tokenizer.decode(generated.flatten().tolist()))


# %%
if __name__ == "__main__":
    main()
