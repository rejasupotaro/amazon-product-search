from typing import Any, Dict

from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer
from amazon_product_search.nlp.extractor import KeywordExtractor


class Analyzer:
    def __init__(self, text_fields: list[str]):
        self.text_fields = text_fields
        self.tokenizer = Tokenizer()
        self.extractor = KeywordExtractor()

    def _normalize(self, s: str) -> str:
        s = normalize_doc(s)
        tokens = self.tokenizer.tokenize(s)
        return " ".join(tokens)

    def analyze(self, product: Dict[str, Any]) -> Dict[str, Any]:
        for field in self.text_fields:
            if field not in product:
                continue

            normalized_text = self._normalize(product[field])
            product[field] = normalized_text
            product["product_yake"] = self.extractor.apply_yake(normalized_text)
            product["product_position_rank"] = self.extractor.apply_position_rank(normalized_text)
            product["product_multipartite_rank"] = self.extractor.apply_multipartite_rank(normalized_text)
        return product
