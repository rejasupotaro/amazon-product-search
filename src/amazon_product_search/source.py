from typing import Literal, Optional

import polars as pl

from amazon_product_search.constants import DATA_DIR

Locale = Literal["jp", "us", "es"]


def load_products(
    locale: Optional[Locale] = None, nrows: int = -1, data_dir: str = DATA_DIR
) -> pl.DataFrame:
    if locale:
        filename = f"{data_dir}/products_{locale}.parquet"
    else:
        filename = f"{data_dir}/raw/shopping_queries_dataset_products.parquet"
    if nrows > 0:
        return pl.read_parquet(filename).head(nrows)
    else:
        return pl.read_parquet(filename)


def load_labels(
    locale: Optional[Locale] = None, nrows: int = -1, data_dir: str = DATA_DIR
) -> pl.DataFrame:
    if locale:
        filename = f"{data_dir}/examples_{locale}.parquet"
    else:
        filename = f"{data_dir}/raw/shopping_queries_dataset_examples.parquet"
    if nrows > 0:
        return pl.read_parquet(filename).head(nrows)
    else:
        return pl.read_parquet(filename)


def load_sources(data_dir: str = DATA_DIR) -> pl.DataFrame:
    filename = f"{data_dir}/raw/shopping_queries_dataset_sources.csv"
    return pl.read_csv(filename).to_pandas()


def load_merged(
    locale: Locale, nrows: int = -1, data_dir: str = DATA_DIR
) -> pl.DataFrame:
    filename = f"{data_dir}/merged_{locale}.parquet"
    if nrows > 0:
        return pl.read_parquet(filename).head(nrows)
    else:
        return pl.read_parquet(filename)


def merge_and_split(data_dir: str = DATA_DIR):
    """Load raw dataset and split it by locale for convenience.

    Raw data files need to be downloaded in the appropriate location in advance.
    - data/raw/shopping_queries_dataset_products.parquet
    - data/raw/shopping_queries_dataset_examples.parquet
    - data/raw/shopping_queries_dataset_sources.parquet

    Then, each file will be processed and saved as shown below.
    - data/products_es.parquet
    - data/products_jp.parquet
    - data/products_us.parquet
    - data/examples_es.parquet
    - data/examples_jp.parquet
    - data/examples_us.parquet
    - data/merged_es.parquet
    - data/merged_jp.parquet
    - data/merged_us.parquet
    """
    print("Load product catalogue")
    products_df = pl.from_pandas(load_products())
    print("Load relevance judgements")
    labels_df = pl.from_pandas(load_labels())
    print("Load sources")
    sources_df = pl.from_pandas(load_sources())
    print("Merge datasets")
    merged_df = labels_df.join(
        products_df, how="left", on=["product_id", "product_locale"]
    ).join(sources_df, how="left", on=["query_id"])

    locales = labels_df["product_locale"].unique().to_list()
    print(f"The dataset contains locales: {locales}")

    print("Split catalogue dataset by locale")
    for locale in locales:
        filtered_df = products_df.filter(pl.col("product_locale") == locale)
        filepath = f"{data_dir}/products_{locale}.parquet"
        filtered_df.write_parquet(filepath)
        print(
            f"Catalog dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}"
        )

    print("Split judgement dataset by locale")
    for locale in locales:
        filtered_df = labels_df.filter(pl.col("product_locale") == locale)
        filepath = f"{data_dir}/examples_{locale}.parquet"
        filtered_df.write_parquet(filepath)
        print(
            f"Label dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}"
        )

    print("Split merged dataset by locale")
    for locale in locales:
        filtered_df = merged_df.filter(pl.col("product_locale") == locale)
        filepath = f"{data_dir}/merged_{locale}.parquet"
        filtered_df.write_parquet(filepath)
        print(
            f"Merged dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}"
        )
