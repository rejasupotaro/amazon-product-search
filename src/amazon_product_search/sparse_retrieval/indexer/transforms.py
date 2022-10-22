from typing import Any, Iterator

import apache_beam as beam

from amazon_product_search.nlp.analyzer import Analyzer
from amazon_product_search.nlp.tokenizer import Tokenizer


class AnalyzeFn(beam.DoFn):
    def __init__(self, text_fields: list[str]):
        self.analyzer = Analyzer(text_fields)

    def setup(self):
        self.tokenizer = Tokenizer()

    def process(self, product: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield self.analyzer.analyze(product)
