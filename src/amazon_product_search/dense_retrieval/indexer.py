import pandas as pd
from annoy import AnnoyIndex

from amazon_product_search.constants import MODELS_DIR
from amazon_product_search.dense_retrieval.encoder import Encoder
from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.source import load_products


def preprocess(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(normalize_doc)
    return df


def load_dataset(locale: str) -> pd.DataFrame:
    products_df = load_products(locale, nrows=1000)
    products_df.fillna("", inplace=True)
    products_df = preprocess(products_df, columns=["product_title", "product_brand"])
    products_df["product"] = products_df["product_title"] + " " + products_df["product_brand"]
    return products_df


def index(locale: str):
    print("Load dataset")
    products_df = load_dataset(locale)

    print("Encode products")
    encoder = Encoder()
    product_vectors = encoder.encode(products_df["product"].tolist())

    print("Index products")
    t = AnnoyIndex(f=768, metric="dot")
    for row, vector in zip(products_df.to_dict("records"), product_vectors):
        t.add_item(row["index"], vector)
    t.build(n_trees=10)

    print("Save index")
    t.save(f"{MODELS_DIR}/products.ann")
    print("Done")


if __name__ == "__main__":
    index(locale="jp")
