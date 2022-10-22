import logging
from typing import Any, Dict, Iterator, List

import apache_beam as beam

from amazon_product_search.dense_retrieval.encoder import Encoder


class EncodeFn(beam.DoFn):
    def __init__(self):
        pass

    def setup(self):
        self.encoder = Encoder()

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        text = product["product_title"] + " " + product["product_brand"]
        product_vectors = self.encoder.encode([text], show_progress_bar=False)
        product["product_vector"] = product_vectors[0]
        yield product


class BatchEncodeFn(beam.DoFn):
    def __init__(self):
        pass

    def setup(self):
        self.encoder = Encoder()

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        logging.info(f"Encoding products with a batch size: {len(products)}")
        texts = [product["product_title"] + " " + product["product_brand"] for product in products]
        product_vectors = self.encoder.encode([texts], show_progress_bar=False)
        for product, product_vector in zip(products, product_vectors):
            product["product_vector"] = product_vector
            yield product
