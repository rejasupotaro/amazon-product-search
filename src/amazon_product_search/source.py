import pandas as pd

from amazon_product_search.constants import DATA_DIR


def load_products(locale: str = "", nrows: int = -1) -> pd.DataFrame:
    if locale:
        filename = f"{DATA_DIR}/products_{locale}.parquet"
    else:
        filename = f"{DATA_DIR}/raw/shopping_queries_dataset_products.parquet"
    if nrows > 0:
        return pd.read_parquet(filename).head(nrows)
    else:
        return pd.read_parquet(filename)


def load_labels(locale: str = "", nrows: int = -1) -> pd.DataFrame:
    if locale:
        filename = f"{DATA_DIR}/examples_{locale}.parquet"
    else:
        filename = f"{DATA_DIR}/raw/shopping_queries_dataset_examples.parquet"
    if nrows > 0:
        return pd.read_parquet(filename).head(nrows)
    else:
        return pd.read_parquet(filename)


def load_sources() -> pd.DataFrame:
    filename = f"{DATA_DIR}/raw/shopping_queries_dataset_sources.parquet"
    return pd.read_parquet(filename)


def load_merged(locale: str, nrows: int = -1) -> pd.DataFrame:
    filename = f"{DATA_DIR}/merged_{locale}.parquet"
    if nrows > 0:
        return pd.read_parquet(filename).head(nrows)
    else:
        return pd.read_parquet(filename)


def merge_and_split():
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
    products_df = load_products()
    print("Load relevance judgements")
    labels_df = load_labels()
    print("Load sources")
    sources_df = load_sources()
    print("Merge datasets")
    merged_df = labels_df.merge(products_df, how="left", on=["product_id", "product_locale"]).merge(
        sources_df, how="left", on=["query_id"]
    )

    locales = labels_df["product_locale"].unique()
    print(f"The dataset contains locales: {locales}")

    print("Split catalogue dataset by locale")
    for locale in locales:
        filtered_df = products_df[products_df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/products_{locale}.parquet"
        filtered_df.to_parquet(filepath, index=False)
        print(f"A catalog dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")

    print("Split judgement dataset by locale")
    for locale in locales:
        filtered_df = labels_df[labels_df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/examples_{locale}.parquet"
        filtered_df.to_parquet(filepath, index=False)
        print(f"A label dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")

    print("Split merged dataset by locale")
    for locale in locales:
        filtered_df = merged_df[merged_df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/merged_{locale}.parquet"
        filtered_df.to_parquet(filepath, index=False)
        print(f"A merged dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")
