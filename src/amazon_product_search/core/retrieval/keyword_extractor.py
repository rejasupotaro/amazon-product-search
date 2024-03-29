from keybert import KeyBERT

from amazon_product_search.constants import HF


class KeywordExtractor:
    def __init__(self) -> None:
        self._keybert = KeyBERT(HF.JP_SBERT_MEAN)

    def apply_keybert(self, text: str) -> list[tuple[str, float]]:
        return self._keybert.extract_keywords(text, top_n=10)
