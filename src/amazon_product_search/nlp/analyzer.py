from typing import Any, Dict

from amazon_product_search.nlp.extractor import KeywordExtractor
from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer


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
        return product
