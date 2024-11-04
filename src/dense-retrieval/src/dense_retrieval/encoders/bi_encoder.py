from torch import Tensor
from torch.nn import Module, functional

from dense_retrieval.encoders.text_encoder import (
    TextEncoder,
    Tokenizer,
)


class QueryEncoder(Module):
    def __init__(
        self,
        text_encoder: TextEncoder,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        text_emb = self.text_encoder(tokens)
        return text_emb


class ProductEncoder(Module):
    def __init__(
        self,
        text_encoder: TextEncoder,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        text_emb = self.text_encoder(tokens)
        return text_emb


class BiEncoder(Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        query_encoder: QueryEncoder,
        product_encoder: ProductEncoder,
        criteria: Module,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.query_encoder = query_encoder
        self.product_encoder = product_encoder
        self.criteria = criteria

    @staticmethod
    def compute_score(query: Tensor, doc: Tensor) -> Tensor:
        return functional.cosine_similarity(query, doc)

    def forward(
        self,
        query: dict[str, Tensor],
        pos_doc: dict[str, Tensor],
        neg_doc: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        query_vec = self.query_encoder(query)
        pos_doc_vec = self.product_encoder(pos_doc)
        neg_doc_vec = self.product_encoder(neg_doc)
        loss: Tensor = self.criteria(query_vec, pos_doc_vec, neg_doc_vec)
        pos_score = self.compute_score(query_vec, pos_doc_vec)
        neg_score = self.compute_score(query_vec, neg_doc_vec)
        acc = (pos_score > neg_score).float()
        return (loss.mean(), acc.mean())
