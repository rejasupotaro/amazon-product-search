import pandas as pd

from amazon_product_search.constants import DATA_DIR


def _split_products_by_locale():
    filename = "product_catalogue-v0.3"
    filepath = f"{DATA_DIR}/{filename}.csv.zip"
    print(f"Load catalog from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.reset_index()

    locales = df["product_locale"].unique()
    print(f"The dataset contains locales: {locales}")

    for locale in locales:
        filtered_df = df[df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/{filename}_{locale}.csv.zip"
        filtered_df.to_csv(filepath, index=False)
        print(f"A catalog dataset (locale: {locale}) containing {len(filtered_df)} rows is saved to {filepath}")


def _split_labels_by_locale():
    filename = "train-v0.3"
    filepath = f"{DATA_DIR}/{filename}.csv.zip"
    print(f"Load labels from {filepath}...")
    df = pd.read_csv(filepath)

    locales = df["query_locale"].unique()
    print(f"The dataset contains locales: {locales}")

    for locale in locales:
        filtered_df = df[df["query_locale"] == locale]
        filepath = f"{DATA_DIR}/{filename}_{locale}.csv.zip"
        filtered_df.to_csv(filepath, index=False)
        print(f"A label dataset (locale: {locale}) containing {len(filtered_df)} rows is saved to {filepath}")


def split_dataset_by_locale():
    _split_products_by_locale()
    _split_labels_by_locale()


def load_products(locale: str, nrows: int = -1) -> pd.DataFrame:
    filename = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows if nrows > 0 else None, engine="python")


def load_labels(locale: str, nrows: int = -1) -> pd.DataFrame:
    filename = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows if nrows > 0 else None, engine="python")
