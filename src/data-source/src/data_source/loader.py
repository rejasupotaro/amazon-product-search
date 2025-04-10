from typing import Literal

import pandas as pd

Locale = Literal["us", "es", "jp"]


def load_examples(data_dir: str, locale: Locale | None = None, nrows: int = -1) -> pd.DataFrame:
    examples_df = pd.read_parquet(f"{data_dir}/shopping_queries_dataset_examples.parquet")
    if locale is not None:
        examples_df = examples_df[examples_df["product_locale"] == locale]
    if nrows > 0:
        examples_df = examples_df[:nrows]
    return examples_df


def load_products(data_dir: str, locale: Locale | None = None, nrows: int = -1) -> pd.DataFrame:
    products_df = pd.read_parquet(f"{data_dir}/shopping_queries_dataset_products.parquet")
    if locale is not None:
        products_df = products_df[products_df["product_locale"] == locale]
    if nrows > 0:
        products_df = products_df[:nrows]
    return products_df


def load_sources(data_dir: str) -> pd.DataFrame:
    sources_df = pd.read_csv(f"{data_dir}/shopping_queries_dataset_sources.csv")
    return sources_df


def load_merged(data_dir: str, locale: Locale | None = None, nrows: int = -1) -> pd.DataFrame:
    examples_df = load_examples(data_dir, locale)
    products_df = load_products(data_dir, locale)
    merged_df = examples_df.merge(
        products_df[["product_id", "product_title"]],
        on="product_id",
        how="left",
    )
    if nrows > 0:
        merged_df = merged_df[:nrows]
    return merged_df
