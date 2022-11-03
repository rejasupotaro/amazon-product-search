from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.nlp.extractor import KeywordExtractor


class ExtractFn(beam.DoFn):
    def __init__(self, text_fields: list[str]):
        self.text_fields = text_fields

    def setup(self):
        self.extractor = KeywordExtractor()

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        for field in self.text_fields:
            if field not in product:
                continue

            text = product[field]
            product["product_yake"] = self.extractor.apply_yake(text)
            product["product_position_rank"] = self.extractor.apply_position_rank(text)
            product["product_multipartite_rank"] = self.extractor.apply_multipartite_rank(text)
        yield product
