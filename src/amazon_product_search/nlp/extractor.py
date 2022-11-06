import pke
from keybert import KeyBERT

from amazon_product_search.nlp.encoder import JA_SBERT


class KeywordExtractor:
    def __init__(self):
        self.yake = pke.unsupervised.YAKE()
        self.position_rank = pke.unsupervised.PositionRank()
        self.multipartite_rank = pke.unsupervised.MultipartiteRank()
        self.keybert = KeyBERT(JA_SBERT)

    def apply_yake(self, text: str) -> list[tuple[str, float]]:
        self.yake.load_document(input=text)
        self.yake.candidate_selection()
        self.yake.candidate_weighting()
        return self.yake.get_n_best(n=10)

    def apply_position_rank(self, text: str) -> list[tuple[str, float]]:
        self.position_rank.load_document(input=text)
        self.position_rank.candidate_selection()
        self.position_rank.candidate_weighting()
        return self.position_rank.get_n_best(n=10)

    def apply_multipartite_rank(self, text: str) -> list[tuple[str, float]]:
        self.multipartite_rank.load_document(input=text)
        self.multipartite_rank.candidate_selection(pos={"NOUN", "PROPN", "ADJ", "NUM"})
        self.multipartite_rank.candidate_weighting()
        return self.multipartite_rank.get_n_best(n=10)

    def apply_keybert(self, text: str) -> list[tuple[str, float]]:
        return self.keybert.extract_keywords(text, top_n=10)
