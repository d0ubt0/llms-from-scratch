# %%
import pandas as pd
from torch.serialization import save


def load_json(file_path):
    return pd.read_json(file_path)


def clean_data(df):
    df = df["Contents"].dropna()
    df = df[df.str.strip() != ""]
    df = df[~df.str.startswith("http")]

    return df.reset_index(drop=True)


def save_cleaned_data(df, output_path="data/output_messages.csv"):
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    file_path = "data/messages.json"
    df = load_json(file_path)
    df = clean_data(df)

    save_cleaned_data(df, output_path="data/output_messages.csv")
