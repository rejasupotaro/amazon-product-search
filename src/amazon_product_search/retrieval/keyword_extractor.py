from keybert import KeyBERT

from amazon_product_search.constants import HF


class KeywordExtractor:
    def __init__(self):
        self._keybert = KeyBERT(HF.JA_SBERT)

    def apply_keybert(self, text: str) -> list[tuple[str, float]]:
        return self._keybert.extract_keywords(text, top_n=10)
