from typing import Any

from annoy import AnnoyIndex

from amazon_product_search.constants import MODELS_DIR
from amazon_product_search.dense_retrieval.encoder import Encoder
from amazon_product_search.models.search import Response, Result
from amazon_product_search.source import load_products


class Retriever:
    def __init__(self):
        self.encoder = Encoder()
        self.t = AnnoyIndex(f=768, metric="dot")
        self.t.load(f"{MODELS_DIR}/products.ann")

        products_df = load_products(locale="jp", nrows=1000)
        products_df.fillna("", inplace=True)
        self.index_to_product: dict[str, Any] = {}
        for row in products_df.to_dict("records"):
            self.index_to_product[row["index"]] = row

    def search(self, query: str, top_k: int) -> Response:
        vector = self.encoder.encode(query, show_progress_bar=False)
        product_indices, distances = self.t.get_nns_by_vector(vector, n=top_k, include_distances=True)
        results = []
        for product_index, distance in zip(product_indices, distances):
            if product_index in self.index_to_product:
                product = self.index_to_product[product_index]
            else:
                product = {"id": product_index, "product_title": "NOT_FOUND"}
            results.append(
                Result(
                    product=product,
                    score=distance,
                )
            )
        return Response(results=results, total_hits=top_k)
