# %%
import torch
import torch.nn as nn

# %%
config = {
    "vocab_size": 50257,
    "dim_emb": 768,
    "context_size": 1028,
    "dropout_rate": 0.1,
    "bias": False,
    "n_layers": 12,
    "n_heads": 12,
}


# %%
class DummyTransformerModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, inputs):
        return inputs


# %%
class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
        self.epsilon = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (inputs - mean) / torch.sqrt(std + self.epsilon)
        return self.weight * normalized + self.bias


# %%
class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config["vocab_size"], config["dim_emb"])
        self.pos_emb = nn.Embedding(config["context_size"], config["dim_emb"])
        self.dropout_layer = nn.Dropout(config["dropout_rate"])

        self.transformers = nn.Sequential(
            *[DummyTransformerModel(config) for i in range(config["n_layers"])]
        )

        self.layer_norm = DummyLayerNorm(config["dim_emb"])
        self.final_layer = nn.Linear(config["dim_emb"], config["vocab_size"])

    def forward(self, inputs):
        batch, tokens_len = inputs.shape

        tokens_embeds = self.token_emb(inputs)
        pos_embeds = self.pos_emb(torch.arange(tokens_len, device=inputs.device))

        embeds = tokens_embeds + pos_embeds

        embeds = self.dropout_layer(embeds)
        embeds = self.transformers(embeds)
        embeds = self.layer_norm(embeds)
        logits = self.final_layer(embeds)

        return logits


# %%
## Fake Flow
import tiktoken

tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model("gpt2")

# %%

texts = ["ab ab", "ac ac"]


batch = []
for text in texts:
    batch.append(torch.tensor(tokenizer.encode(text)))

batch = torch.stack(batch)
print(batch)

# %%
gpt = DummyGPTModel(config)

# %%
outputs = gpt(batch)
print(outputs.shape)
print(outputs)
# %%


ln = DummyLayerNorm(5)
test = torch.rand((10, 5))
outputs = ln(test)
print(outputs.shape)
print(outputs)
print(outputs.mean(dim=-1, keepdim=True))
print(outputs.var(dim=-1, unbiased=False, keepdim=True))
# %%


class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config["dim_emb"], 4 * config["dim_emb"]),
            nn.ReLU(),
            nn.Linear(4 * config["dim_emb"], config["dim_emb"]),
        )

    def forward(self, x):
        return self.layer(x)
