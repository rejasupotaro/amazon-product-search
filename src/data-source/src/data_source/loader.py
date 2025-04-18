from pathlib import Path
from typing import Literal

import pandas as pd

Locale = Literal["us", "es", "jp"]


def find_data_dir() -> str:
    """Find the data directory for the package.

    If I want to make it more robust to changes, I could use
    ```
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        data_dir = parent / "data"
        if data_dir.exists():
            return data_dir
    raise FileNotFoundError("Data directory not found.")
    ```

    However, this is a simple solution that works for the current structure of the package.
    """
    package_root = Path(__file__).resolve().parents[2]
    return str(package_root / "data")


def load_examples(locale: Locale | None = None, nrows: int = -1, data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    examples_df = pd.read_parquet(f"{data_dir}/shopping_queries_dataset_examples.parquet")
    if locale is not None:
        examples_df = examples_df[examples_df["product_locale"] == locale]
    if nrows > 0:
        examples_df = examples_df[:nrows]
    return examples_df


def load_products(locale: Locale | None = None, nrows: int = -1, data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    products_df = pd.read_parquet(f"{data_dir}/shopping_queries_dataset_products.parquet")
    if locale is not None:
        products_df = products_df[products_df["product_locale"] == locale]
    if nrows > 0:
        products_df = products_df[:nrows]
    return products_df


def load_sources(data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = find_data_dir()
    sources_df = pd.read_csv(f"{data_dir}/shopping_queries_dataset_sources.csv")
    return sources_df


def load_merged(locale: Locale | None = None, nrows: int = -1, data_dir: str | None = None) -> pd.DataFrame:
    examples_df = load_examples(locale=locale, data_dir=data_dir)
    products_df = load_products(locale=locale, data_dir=data_dir)
    merged_df = examples_df.merge(
        products_df[["product_id", "product_title"]],
        on="product_id",
        how="left",
    )
    if nrows > 0:
        merged_df = merged_df[:nrows]
    return merged_df
