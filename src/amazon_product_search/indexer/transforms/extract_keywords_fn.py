from typing import Any, Dict, Iterator, Tuple

import apache_beam as beam

from amazon_product_search.retrieval.keyword_extractor import KeywordExtractor


class ExtractKeywordsFn(beam.DoFn):
    def setup(self):
        self._extractor = KeywordExtractor()

    @staticmethod
    def convert_results_to_text(results: list[tuple[str, float]]) -> str:
        return " ".join([keyword for keyword, score in results])

    def process(self, product: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, str]]]:
        result: Dict[str, str] = {}

        text = product["product_description"] + " " + product["product_bullet_point"]
        text = text.strip()
        if not text:
            yield product["product_id"], result
            return

        result["product_description_keybert"] = self.convert_results_to_text(self._extractor.apply_keybert(text))
        yield product["product_id"], result
