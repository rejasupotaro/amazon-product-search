from typing import Any, Dict, Iterator, Tuple

import apache_beam as beam
from apache_beam.utils.shared import Shared

from amazon_product_search.indexer.transforms.weak_reference import WeakReference
from amazon_product_search.keyword_extractor import KeywordExtractor


class ExtractKeywordsFn(beam.DoFn):
    def __init__(self, shared_handle: Shared):
        self._shared_handle = shared_handle

    def setup(self):
        def initialize_extractor():
            # Load a potentially large model in memory. Executed once per process.
            return WeakReference[KeywordExtractor](KeywordExtractor())

        self._weak_reference: WeakReference[KeywordExtractor] = self._shared_handle.acquire(initialize_extractor)

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

        result["product_description_yake"] = self.convert_results_to_text(self._weak_reference.ref.apply_yake(text))
        result["product_description_position_rank"] = self.convert_results_to_text(
            self._weak_reference.ref.apply_position_rank(text)
        )
        result["product_description_multipartite_rank"] = self.convert_results_to_text(
            self._weak_reference.ref.apply_multipartite_rank(text)
        )
        result["product_description_keybert"] = self.convert_results_to_text(
            self._weak_reference.ref.apply_keybert(text)
        )
        yield product["product_id"], result
