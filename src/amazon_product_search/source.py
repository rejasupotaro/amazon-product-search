import pandas as pd

from amazon_product_search.constants import DATA_DIR


def load_products(locale: str = "", nrows: int = -1) -> pd.DataFrame:
    if locale:
        filename = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    else:
        filename = f"{DATA_DIR}/raw/product_catalogue-v0.3.csv.zip"
    return pd.read_csv(filename, nrows=nrows if nrows > 0 else None, engine="python")


def load_labels(locale: str = "", nrows: int = -1) -> pd.DataFrame:
    if locale:
        filename = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
    else:
        filename = f"{DATA_DIR}/raw/train-v0.3.csv.zip"
    return pd.read_csv(filename, nrows=nrows if nrows > 0 else None, engine="python")


def merge_and_split():
    """Load raw dataset and split it by locale for convenience."""
    print("Load catalogue dataset")
    products_df = load_products()
    print("Load judgement dataset")
    labels_df = load_labels()
    print("Merge two datasets")
    merged_df = labels_df.merge(products_df, on="product_id", how="left")

    locales = labels_df["query_locale"].unique()
    print(f"The dataset contains locales: {locales}")

    print("Split catalogue dataset by locale")
    for locale in locales:
        filtered_df = products_df[products_df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
        filtered_df.to_csv(filepath, index=False)
        print(f"A catalog dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")

    print("Split judgement dataset by locale")
    for locale in locales:
        filtered_df = labels_df[labels_df["query_locale"] == locale]
        filepath = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
        filtered_df.to_csv(filepath, index=False)
        print(f"A label dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")

    print("Split merged dataset by locale")
    for locale in locales:
        filtered_df = merged_df[merged_df["query_locale"] == locale]
        filepath = f"{DATA_DIR}/merged-v0.3_{locale}.csv.zip"
        filtered_df.to_csv(filepath, index=False)
        print(f"A merged dataset (locale: {locale}) containing {len(filtered_df)} rows was saved to {filepath}")
