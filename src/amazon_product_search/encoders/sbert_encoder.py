from sentence_transformers import SentenceTransformer
from torch import Tensor

from amazon_product_search.constants import HF


class SBERTEncoder:
    def __init__(self, model_name: str = HF.JP_SBERT):
        self.sentence_transformer = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> Tensor:
        return self.sentence_transformer.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
