from typing import Any, Iterator

import pandas as pd

from amazon_product_search.dense_retrieval.encoder import Encoder
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.nlp.analyzer import Analyzer
from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.source import load_products


def preprocess(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(normalize_doc)
    return df


def load_dataset(locale: str) -> pd.DataFrame:
    products_df = load_products(locale, nrows=10)
    products_df.fillna("", inplace=True)
    products_df = preprocess(products_df, columns=["product_title", "product_brand"])
    products_df["product"] = products_df["product_title"] + " " + products_df["product_brand"]
    return products_df


def encode_docs(docs: list[str]):
    encoder = Encoder()
    vectors = encoder.encode(docs)
    return vectors


def index_docs(docs: Iterator[tuple[dict[str, Any], Any]]):
    analyzer = Analyzer(text_fields=["product_title", "product_description", "product_bullet_point"])
    es_client = EsClient()
    index_name = "products_jp"
    for row, vector in docs:
        doc = analyzer.analyze(row)
        doc["product_vector"] = vector
        doc_id = doc["product_id"]
        es_client.index_doc(index_name=index_name, doc=doc, doc_id=doc_id)


def run(locale: str):
    print("Load dataset")
    products_df = load_dataset(locale)

    print("Encode products")
    product_vectors = encode_docs(products_df["product"].tolist())

    print("Index products")
    index_docs(zip(products_df.to_dict("records"), product_vectors))

    print("Done")


if __name__ == "__main__":
    run(locale="jp")
