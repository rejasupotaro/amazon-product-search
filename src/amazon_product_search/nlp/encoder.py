from sentence_transformers import SentenceTransformer
from torch import Tensor

from amazon_product_search.constants import HF


class Encoder:
    def __init__(self, model_name: str = HF.JP_SBERT):
        self.embedder = SentenceTransformer(model_name)

    def encode(self, texts: list[str], show_progress_bar: bool = False, convert_to_tensor: bool = False) -> Tensor:
        return self.embedder.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
