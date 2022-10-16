from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer


class AnalyzeFn(beam.DoFn):
    def __init__(self, text_fields: list[str]):
        self.text_fields = text_fields

    def setup(self):
        self.tokenizer = Tokenizer()

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        for field in self.text_fields:
            s = product[field]
            s = normalize_doc(s)
            tokens = self.tokenizer.tokenize(s)
            product[field] = " ".join(tokens)
        yield product
