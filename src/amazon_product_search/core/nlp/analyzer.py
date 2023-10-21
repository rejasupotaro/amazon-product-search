from typing import Any, Dict, cast

from amazon_product_search.core.nlp.normalizer import normalize_doc
from amazon_product_search.core.nlp.tokenizers import EnglishTokenizer, JapaneseTokenizer, Tokenizer
from amazon_product_search.core.source import Locale


class Analyzer:
    def __init__(self, text_fields: list[str], locale: Locale) -> None:
        self.text_fields = text_fields
        self.tokenizer: Tokenizer = {
            "us": EnglishTokenizer,
            "jp": JapaneseTokenizer,
        }[locale]()

    def _normalize(self, s: str) -> str:
        s = normalize_doc(s)
        tokens = self.tokenizer.tokenize(s)
        return " ".join(cast(list, tokens))

    def analyze(self, product: Dict[str, Any]) -> Dict[str, Any]:
        for field in self.text_fields:
            if field not in product:
                continue

            normalized_text = self._normalize(product[field])
            product[field] = normalized_text
        return product
