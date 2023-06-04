from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.nlp.analyzer import Analyzer


class AnalyzeFn(beam.DoFn):
    def __init__(self, text_fields: list[str]) -> None:
        self.text_fields = text_fields

    def setup(self) -> None:
        self.analyzer = Analyzer(self.text_fields)

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield self.analyzer.analyze(product)
