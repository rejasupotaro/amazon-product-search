from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.core.nlp.analyzer import Analyzer
from amazon_product_search.core.source import Locale


class AnalyzeFn(beam.DoFn):
    def __init__(self, text_fields: list[str], locale: Locale) -> None:
        self.text_fields = text_fields
        self.locale = locale

    def setup(self) -> None:
        self.analyzer = Analyzer(self.text_fields, self.locale)

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield self.analyzer.analyze(product)
