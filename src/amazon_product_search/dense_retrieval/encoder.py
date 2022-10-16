from sentence_transformers import SentenceTransformer
from torch import Tensor


class Encoder:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)

    def encode(self, texts: list[str], convert_to_tensor=False) -> Tensor:
        vectors = self.embedder.encode(texts, show_progress_bar=True, convert_to_tensor=convert_to_tensor)
        return vectors
