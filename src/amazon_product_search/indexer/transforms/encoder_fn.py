from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.dense_retrieval.encoder import Encoder


class EncoderFn(beam.DoFn):
    def __init__(self):
        pass

    def setup(self):
        self.encoder = Encoder()

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        text = product["product_title"] + " " + product["product_brand"]
        product_vectors = self.encoder.encode([text], show_progress_bar=False)
        product["product_vector"] = product_vectors[0]
        yield product
