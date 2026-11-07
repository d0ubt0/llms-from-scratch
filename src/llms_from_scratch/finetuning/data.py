"""Datos para el fine-tuning de clasificación de SMS spam.

El dataset original tiene 5 572 mensajes (4 825 ham / 747 spam). Aquí se
eliminan duplicados, se hace un split estratificado y se construyen
``DataLoader`` con padding fijo o dinámico.

Uso rápido::

    uv run python -m llms_from_scratch.finetuning.data
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from llms_from_scratch.tokenizer import EOS_TOKEN, TextTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = PROJECT_ROOT / "sms_spam_collection.zip"
EXTRACTED_PATH = PROJECT_ROOT / "sms_spam_collection"
DATA_FILE_PATH = EXTRACTED_PATH / "SMSSpamCollection.tsv"

# Nombres antiguos que algunos scripts importaban directamente.
url = URL
zip_path = ZIP_PATH
extracted_path = EXTRACTED_PATH
data_file_path = DATA_FILE_PATH

LABEL_TO_ID = {"ham": 0, "spam": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

# GPT-2 usa <|endoftext|> como token de padding.
DEFAULT_PAD_TOKEN_ID = 50256

PaddingMode = Literal["dynamic", "fixed"]
BalanceMode = Literal["none", "undersample"]


def download_and_unzip_spam_data(
    url: str = URL,
    zip_path: str | Path = ZIP_PATH,
    extracted_path: str | Path = EXTRACTED_PATH,
    data_file_path: str | Path = DATA_FILE_PATH,
) -> None:
    """Descarga y descomprime el dataset si todavía no está en disco."""
    zip_path = Path(zip_path)
    extracted_path = Path(extracted_path)
    data_file_path = Path(data_file_path)

    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.mkdir(parents=True, exist_ok=True)

    # Descarga el archivo zip
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    # Descomprime el contenido
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Renombra el archivo original para añadir la extensión .tsv
    original_file_path = extracted_path / "SMSSpamCollection"
    if original_file_path != data_file_path:
        os.replace(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def load_spam_data(
    data_file_path: str | Path = DATA_FILE_PATH,
) -> pd.DataFrame:
    """Carga el TSV en un DataFrame con columnas ``label`` y ``message``."""
    data_file_path = Path(data_file_path)
    if not data_file_path.exists():
        raise FileNotFoundError(
            f"{data_file_path} does not exist. Please download the data first."
        )

    df = pd.read_csv(
        data_file_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        dtype={"label": str, "message": str},
    )
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].fillna("").str.strip()
    df = df[df["message"] != ""].reset_index(drop=True)

    unknown = sorted(set(df["label"]) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Etiquetas desconocidas en el dataset: {unknown}")

    return df


def deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina mensajes repetidos para evitar fuga de datos entre splits."""
    return df.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)


def add_label_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["label_id"] = result["label"].map(LABEL_TO_ID)
    if result["label_id"].isna().any():
        unknown = sorted(result.loc[result["label_id"].isna(), "label"].unique())
        raise ValueError(f"Etiquetas desconocidas: {unknown}")
    result["label_id"] = result["label_id"].astype(int)
    return result


def balance_dataset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Balancea por undersampling de la clase mayoritaria (ham)."""
    spam_df = df[df["label"] == "spam"]
    ham_df = df[df["label"] == "ham"]

    if spam_df.empty or ham_df.empty:
        return df.reset_index(drop=True)

    n = min(len(spam_df), len(ham_df))
    spam_df = spam_df.sample(n=n, random_state=random_state)
    ham_df = ham_df.sample(n=n, random_state=random_state)

    balanced_df = (
        pd.concat([spam_df, ham_df])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aleatorio simple (se mantiene por compatibilidad)."""
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_index = int(len(shuffled_df) * (1 - test_size))
    train_df = shuffled_df.iloc[:split_index].reset_index(drop=True)
    test_df = shuffled_df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide manteniendo la proporción ham/spam en los tres conjuntos."""
    if not 0 <= val_size < 1 or not 0 <= test_size < 1:
        raise ValueError("val_size y test_size deben estar en [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size debe ser menor que 1.")

    if "label" not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'label'.")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for label, group in df.groupby("label", sort=True):
        if len(group) < 2:
            raise ValueError(f"La clase {label!r} tiene menos de 2 ejemplos.")

        shuffled = group.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(shuffled)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))

        # Garantiza al menos un ejemplo de train por clase.
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - n_test - 1)

        test_parts.append(shuffled.iloc[:n_test])
        val_parts.append(shuffled.iloc[n_test : n_test + n_val])
        train_parts.append(shuffled.iloc[n_test + n_val :])

    def concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(parts)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    return concat(train_parts), concat(val_parts), concat(test_parts)


def prepare_splits(
    df: pd.DataFrame,
    balance: BalanceMode = "none",
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deduplica, añade ``label_id``, divide y opcionalmente balancea train.

    En la variante ``undersample`` solo se balancea el conjunto de train; val
    y test conservan la distribución real para que las métricas sean
    comparables entre configuraciones.
    """
    if balance not in {"none", "undersample"}:
        raise ValueError(f"balance debe ser 'none' o 'undersample', no {balance!r}.")

    df = deduplicate_dataset(df)
    df = add_label_ids(df)
    train_df, val_df, test_df = stratified_split(
        df, val_size=val_size, test_size=test_size, random_state=random_state
    )

    if balance == "undersample":
        train_df = balance_dataset(train_df, random_state=random_state)
        train_df = add_label_ids(train_df)

    for name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split.empty:
            raise ValueError(f"El split {name} quedó vacío; ajusta los tamaños.")

    return train_df, val_df, test_df


class SpamDataset(Dataset):
    """Tokeniza los mensajes una sola vez y devuelve (ids, etiqueta)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TextTokenizer,
        max_length: int = 128,
        pad_token_id: int | None = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length debe ser mayor que 0.")
        if "label_id" not in df.columns:
            df = add_label_ids(df)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = df["message"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()

        if pad_token_id is None:
            pad_token_id = tokenizer.special_token_id(EOS_TOKEN)
        self.pad_token_id = (
            DEFAULT_PAD_TOKEN_ID if pad_token_id is None else int(pad_token_id)
        )

        self.token_ids = [self._encode(text) for text in self.texts]

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)[: self.max_length]
        if not ids:
            ids = [self.pad_token_id]
        return ids

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.token_ids[index], self.labels[index]


def collate_batch(
    batch: list[tuple[list[int], int]],
    *,
    pad_token_id: int,
    max_length: int,
    padding: PaddingMode = "dynamic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construye un batch con right padding.

    Devuelve ``input_ids``, ``lengths`` (longitud real de cada secuencia) y
    ``labels``. Las longitudes son necesarias para el pooling del último token
    real cuando el padding es dinámico.
    """
    if padding not in {"dynamic", "fixed"}:
        raise ValueError(f"padding debe ser 'dynamic' o 'fixed', no {padding!r}.")

    sequences, labels = zip(*batch, strict=True)
    lengths = torch.tensor(
        [min(len(seq), max_length) for seq in sequences], dtype=torch.long
    )

    if padding == "dynamic":
        target_length = int(lengths.max().item())
    else:
        target_length = max_length
    target_length = max(1, min(target_length, max_length))

    input_ids = torch.full(
        (len(batch), target_length), pad_token_id, dtype=torch.long
    )
    for row, seq in enumerate(sequences):
        length = int(lengths[row].item())
        input_ids[row, :length] = torch.tensor(seq[:length], dtype=torch.long)

    return input_ids, lengths, torch.tensor(labels, dtype=torch.long)


def make_collate_fn(
    pad_token_id: int,
    max_length: int,
    padding: PaddingMode = "dynamic",
):
    return partial(
        collate_batch,
        pad_token_id=pad_token_id,
        max_length=max_length,
        padding=padding,
    )


def make_dataloader(
    df: pd.DataFrame,
    tokenizer: TextTokenizer,
    *,
    pad_token_id: int | None = None,
    max_length: int = 128,
    padding: PaddingMode = "dynamic",
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = SpamDataset(
        df,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(
            dataset.pad_token_id, max_length=max_length, padding=padding
        ),
        generator=generator if shuffle else None,
    )


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: TextTokenizer,
    *,
    pad_token_id: int | None = None,
    max_length: int = 128,
    padding: PaddingMode = "dynamic",
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def build(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        return make_dataloader(
            df,
            tokenizer,
            pad_token_id=pad_token_id,
            max_length=max_length,
            padding=padding,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
        )

    return (
        build(train_df, shuffle=True),
        build(val_df, shuffle=False),
        build(test_df, shuffle=False),
    )


if __name__ == "__main__":
    from llms_from_scratch.tokenizer import get_tokenizer

    download_and_unzip_spam_data()
    df = load_spam_data()
    train_df, val_df, test_df = prepare_splits(df, balance="none")

    print(f"Dataset: {len(df):,} mensajes | {df['label'].value_counts().to_dict()}")
    print(
        f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    print("Train por clase:", train_df["label"].value_counts().to_dict())
    print("Val por clase:", val_df["label"].value_counts().to_dict())
    print("Test por clase:", test_df["label"].value_counts().to_dict())

    tokenizer = get_tokenizer("gpt2")
    train_loader, _, _ = make_dataloaders(
        train_df, val_df, test_df, tokenizer, max_length=128, padding="dynamic"
    )
    input_ids, lengths, labels = next(iter(train_loader))
    print(
        "Primer batch:",
        tuple(input_ids.shape),
        "lengths:",
        lengths.tolist(),
        "labels:",
        labels.tolist(),
    )
