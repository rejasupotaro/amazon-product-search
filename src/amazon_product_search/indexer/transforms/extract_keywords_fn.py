from typing import Any, Dict, Iterator

import apache_beam as beam

from amazon_product_search.keyword_extractor import KeywordExtractor


class ExtractKeywordsFn(beam.DoFn):
    def setup(self):
        self.extractor = KeywordExtractor()

    @staticmethod
    def convert_results_to_text(results: list[tuple[str, float]]) -> str:
        return " ".join([keyword for keyword, score in results])

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        text = product["product_description"] + " " + product["product_bullet_point"]
        text = text.strip()
        if not text:
            yield product

        product["product_description_yake"] = self.convert_results_to_text(self.extractor.apply_yake(text))
        product["product_description_position_rank"] = self.convert_results_to_text(
            self.extractor.apply_position_rank(text)
        )
        product["product_description_multipartite_rank"] = self.convert_results_to_text(
            self.extractor.apply_multipartite_rank(text)
        )
        product["product_description_keybert"] = self.convert_results_to_text(self.extractor.apply_keybert(text))
        yield product
