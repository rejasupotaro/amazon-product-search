from typing import Any

from annoy import AnnoyIndex

from amazon_product_search.constants import MODELS_DIR
from amazon_product_search.dense_retrieval.encoder import Encoder
from amazon_product_search.source import load_products


class Retriever:
    def __init__(self):
        self.encoder = Encoder(model_name="cl-tohoku/bert-base-japanese-v2")
        self.t = AnnoyIndex(f=768, metric="dot")
        self.t.load(f"{MODELS_DIR}/products.ann")

        products_df = load_products(locale="jp", nrows=1000)
        products_df.fillna("", inplace=True)
        self.id_to_product: dict[str, Any] = {}
        for row in products_df.to_dict("records"):
            self.id_to_product[row["index"]] = row

    def search(self, query: str) -> list[dict[str, Any]]:
        vector = self.encoder.encode(query)
        products = []
        for product_idx in self.t.get_nns_by_vector(vector, 5):
            if product_idx not in self.id_to_product:
                products.append({"id": product_idx, "product_title": "NOT_FOUND"})
                continue
            products.append(self.id_to_product[product_idx])
        return products
