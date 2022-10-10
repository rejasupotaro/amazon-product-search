import pandas as pd

DATA_DIR = "./data"


def load_products(locale: str, nrows: int) -> pd.DataFrame:
    filename = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows)


def load_labels(locale: str, nrows: int) -> pd.DataFrame:
    filename = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows)
