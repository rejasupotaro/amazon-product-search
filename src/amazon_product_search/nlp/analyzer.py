from typing import Any

from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer


class Analyzer:
    def __init__(self, text_fields: list[str]):
        self.text_fields = text_fields
        self.tokenizer = Tokenizer()

    def analyze(self, product: dict[str, Any]) -> dict[str, Any]:
        for field in self.text_fields:
            if field not in product:
                continue
            s = product[field]
            s = normalize_doc(s)
            tokens = self.tokenizer.tokenize(s)
            product[field] = " ".join(tokens)
        return product
