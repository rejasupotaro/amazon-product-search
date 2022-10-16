import pandas as pd
from annoy import AnnoyIndex
from dense_retrieval.encoder import Encoder

from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.source import load_products

DATA_DIR = "data"
MODELS_DIR = "models"


def preprocess(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(normalize_doc)
    return df


def load_dataset(locale: str) -> pd.DataFrame:
    products_df = load_products(locale, nrows=100)
    products_df.fillna("", inplace=True)
    products_df = preprocess(products_df, columns=["product_title", "product_brand"])
    products_df["product"] = products_df["product_title"] + " " + products_df["product_brand"]
    return products_df


def index(locale: str, model_name: str):
    print("Load dataset")
    products_df = load_dataset(locale)

    print("Encode products")
    encoder = Encoder(model_name)
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
    # For English
    # model_name = "cross-encoder/ms-marco-electra-base"
    # model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # model_name = "sentence-transformers/msmarco-roberta-base-v3"

    # For Spanish
    # model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    # model_name = "bertin-project/bertin-roberta-base-spanish"

    # For Japanese
    model_name = "cl-tohoku/bert-base-japanese-v2"
    # model_name = "cl-tohoku/bert-base-japanese-char-v2"
    # model_name = "nlp-waseda/roberta-large-japanese"

    # Multi-lingual
    # model_name = "paraphrase-multilingual-mpnet-base-v2"
    # model_name = "stsb-xlm-r-multilingual"

    index(locale="jp", model_name=model_name)
