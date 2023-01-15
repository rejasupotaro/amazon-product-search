import pke
from keybert import KeyBERT

from amazon_product_search.constants import HF


class KeywordExtractor:
    def __init__(self):
        self._yake = pke.unsupervised.YAKE()
        self._position_rank = pke.unsupervised.PositionRank()
        self._multipartite_rank = pke.unsupervised.MultipartiteRank()
        self._keybert = KeyBERT(HF.JA_SBERT)

    def apply_yake(self, text: str) -> list[tuple[str, float]]:
        self._yake.load_document(input=text)
        self._yake.candidate_selection()
        self._yake.candidate_weighting()
        return self._yake.get_n_best(n=10)

    def apply_position_rank(self, text: str) -> list[tuple[str, float]]:
        self._position_rank.load_document(input=text)
        self._position_rank.candidate_selection()
        self._position_rank.candidate_weighting()
        return self._position_rank.get_n_best(n=10)

    def apply_multipartite_rank(self, text: str) -> list[tuple[str, float]]:
        self._multipartite_rank.load_document(input=text)
        self._multipartite_rank.candidate_selection(pos={"NOUN", "PROPN", "ADJ", "NUM"})
        self._multipartite_rank.candidate_weighting()
        return self._multipartite_rank.get_n_best(n=10)

    def apply_keybert(self, text: str) -> list[tuple[str, float]]:
        return self._keybert.extract_keywords(text, top_n=10)
