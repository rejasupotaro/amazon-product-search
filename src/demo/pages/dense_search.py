from typing import Any

import pandas as pd
import streamlit as st
from annoy import AnnoyIndex

from dense_retrieval.encoder import Encoder

MODELS_DIR = "models"
DATA_DIR = "data"


class Retriever:
    def __init__(self):
        self.encoder = Encoder(model_name="cl-tohoku/bert-base-japanese-v2")
        self.t = AnnoyIndex(f=768, metric="dot")
        self.t.load(f"{MODELS_DIR}/products.ann")

        products_filepath = f"{DATA_DIR}/product_catalogue-v0.3_jp.csv.zip"
        products_df = pd.read_csv(
            products_filepath, usecols=["index", "product_id", "product_title", "product_brand"], nrows=100
        )
        products_df.fillna("", inplace=True)
        self.id_to_product: dict[str, Any] = {}
        for row in products_df.to_dict("records"):
            self.id_to_product[row["index"]] = row

    def search(self, query: str) -> list[dict[str, Any]]:
        vector = self.encoder.encode(query)
        products = []
        for product_idx in self.t.get_nns_by_vector(vector, 5):
            product = self.id_to_product[product_idx]
            products.append(product)
        return products


def draw_products(products: list[Any]):
    for product in products:
        st.write(product)
        st.write("----")


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")
    retriever = Retriever()

    query = st.text_input("Query:")
    if not query:
        return

    products = retriever.search(query)
    draw_products(products)


if __name__ == "__main__":
    main()
