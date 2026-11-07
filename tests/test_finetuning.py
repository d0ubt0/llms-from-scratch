"""Pruebas rápidas de datos y modelo para el fine-tuning (sin descargar GPT-2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from llms_from_scratch.config import Config
from llms_from_scratch.finetuning.data import (
    LABEL_TO_ID,
    SpamDataset,
    collate_batch,
    deduplicate_dataset,
    prepare_splits,
)
from llms_from_scratch.finetuning.model import (
    SpamClassifier,
    configure_trainable,
    count_trainable_parameters,
    resolve_model_name,
)
from llms_from_scratch.finetuning.train import TrainingConfig, save_checkpoint
from llms_from_scratch.llm import LLM


class FakeTokenizer:
    """Tokenizer mínimo para probar el dataset sin depender de tiktoken."""

    def encode(self, text: str) -> list[int]:
        return [1] * len(text)

    def decode(self, tokens: list[int]) -> str:
        return ""

    def special_token_id(self, token: str) -> int | None:
        return 99


def make_dataframe(n_ham: int = 50, n_spam: int = 20) -> pd.DataFrame:
    rows = [{"label": "ham", "message": f"ham message {i}"} for i in range(n_ham)]
    rows += [{"label": "spam", "message": f"spam message {i}"} for i in range(n_spam)]
    return pd.DataFrame(rows)


class DataTests(unittest.TestCase):
    def test_deduplicate(self) -> None:
        df = make_dataframe()
        duplicated = pd.concat([df, df.iloc[:5]], ignore_index=True)
        clean = deduplicate_dataset(duplicated)

        self.assertEqual(len(clean), len(df))
        self.assertEqual(clean["message"].nunique(), len(clean))

    def test_stratified_split_has_no_overlap(self) -> None:
        df = make_dataframe()
        train, val, test = prepare_splits(df, balance="none", random_state=42)

        train_messages = set(train["message"])
        val_messages = set(val["message"])
        test_messages = set(test["message"])

        self.assertFalse(train_messages & val_messages)
        self.assertFalse(train_messages & test_messages)
        self.assertFalse(val_messages & test_messages)
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        self.assertGreater(len(train[train["label"] == "spam"]), 0)
        self.assertGreater(len(val[val["label"] == "spam"]), 0)
        self.assertGreater(len(test[test["label"] == "spam"]), 0)

    def test_undersample_only_affects_train(self) -> None:
        df = make_dataframe(n_ham=200, n_spam=40)
        train, val, test = prepare_splits(
            df, balance="undersample", random_state=42
        )

        self.assertEqual(train["label"].value_counts()["ham"], train["label"].value_counts()["spam"])
        self.assertGreater(
            val["label"].value_counts().get("ham", 0),
            val["label"].value_counts().get("spam", 0),
        )
        self.assertGreater(
            test["label"].value_counts().get("ham", 0),
            test["label"].value_counts().get("spam", 0),
        )

    def test_dynamic_collate(self) -> None:
        batch = [
            ([1, 2, 3], 0),
            ([4, 5, 6, 7, 8], 1),
            ([9], 0),
        ]
        input_ids, lengths, labels = collate_batch(
            batch, pad_token_id=99, max_length=8, padding="dynamic"
        )

        self.assertEqual(tuple(input_ids.shape), (3, 5))
        self.assertEqual(lengths.tolist(), [3, 5, 1])
        self.assertEqual(labels.tolist(), [0, 1, 0])
        self.assertEqual(input_ids[0, 3:].tolist(), [99, 99])
        self.assertEqual(input_ids[2, 1:].tolist(), [99, 99, 99, 99])
        self.assertEqual(input_ids[1, 4].item(), 8)

    def test_fixed_collate(self) -> None:
        batch = [([1, 2], 1), ([3], 0)]
        input_ids, lengths, _ = collate_batch(
            batch, pad_token_id=0, max_length=6, padding="fixed"
        )

        self.assertEqual(tuple(input_ids.shape), (2, 6))
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertEqual(input_ids[0, 2:].tolist(), [0, 0, 0, 0])

    def test_spam_dataset_truncates(self) -> None:
        df = make_dataframe()
        dataset = SpamDataset(
            df,
            FakeTokenizer(),
            max_length=5,
            pad_token_id=99,
        )

        token_ids, label = dataset[0]
        self.assertLessEqual(len(token_ids), 5)
        self.assertEqual(label, LABEL_TO_ID["ham"])

    def test_collate_caps_at_max_length(self) -> None:
        batch = [([1] * 20, 1)]
        input_ids, lengths, _ = collate_batch(
            batch, pad_token_id=0, max_length=8, padding="dynamic"
        )

        self.assertEqual(tuple(input_ids.shape), (1, 8))
        self.assertEqual(lengths.tolist(), [8])


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            emb_dim=16,
            vocab_size=50,
            ff_dim_multiplier=2,
            layers=2,
            att_heads=2,
            context_length=16,
            drop_rate=0.0,
        )
        self.model = SpamClassifier(LLM(self.config), drop_rate=0.0)

    def test_forward_shape(self) -> None:
        input_ids = torch.randint(0, self.config.vocab_size, (2, 6))
        lengths = torch.tensor([6, 3])

        logits = self.model(input_ids, lengths)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(self.model.emb_dim, 16)

    def test_out_projection_is_removed(self) -> None:
        self.assertFalse(hasattr(self.model.backbone, "out_projection"))

    def test_head_scope_only_trains_head(self) -> None:
        configure_trainable(self.model, "head")
        trainable = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("head.") for name in trainable))
        self.assertGreater(count_trainable_parameters(self.model), 0)

    def test_last_block_scope(self) -> None:
        configure_trainable(self.model, "last_block")
        trainable = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        self.assertIn("backbone.transformers.1.attention.wq.weight", trainable)
        self.assertIn("backbone.final_norm.weight", trainable)
        self.assertIn("head.2.weight", trainable)
        self.assertNotIn("backbone.transformers.0.attention.wq.weight", trainable)
        self.assertNotIn("backbone.token_embedding.weight", trainable)

    def test_last_two_blocks_scope(self) -> None:
        configure_trainable(self.model, "last_2_blocks")
        trainable = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        self.assertIn("backbone.transformers.0.attention.wq.weight", trainable)
        self.assertIn("backbone.transformers.1.attention.wq.weight", trainable)
        self.assertNotIn("backbone.token_embedding.weight", trainable)

    def test_full_scope(self) -> None:
        configure_trainable(self.model, "full")
        trainable = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        self.assertIn("backbone.token_embedding.weight", trainable)
        self.assertIn("backbone.transformers.0.attention.wq.weight", trainable)
        self.assertIn("head.2.weight", trainable)

    def test_unfreeze_last_override(self) -> None:
        configure_trainable(self.model, "head", unfreeze_last=0)
        trainable = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(all(name.startswith("head.") for name in trainable))

    def test_resolve_model_name(self) -> None:
        self.assertEqual(resolve_model_name("small"), "gpt2")
        self.assertEqual(resolve_model_name("medium"), "gpt2-medium")
        self.assertEqual(resolve_model_name("gpt2-large"), "gpt2-large")
        with self.assertRaises(ValueError):
            resolve_model_name("mistral")

    def test_checkpoint_stores_only_trainable_parameters(self) -> None:
        configure_trainable(self.model, "head")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            config = TrainingConfig(
                model="small",
                trainable="head",
                out_dir=directory,
            )
            save_checkpoint(path, self.model, self.config, config, 0.5, {})

            payload = torch.load(path, map_location="cpu")
            self.assertEqual(payload["state_format"], "trainable_only")
            self.assertEqual(payload["threshold"], 0.5)
            self.assertTrue(payload["model"])
            self.assertTrue(
                all(name.startswith("head.") for name in payload["model"])
            )


if __name__ == "__main__":
    unittest.main()
