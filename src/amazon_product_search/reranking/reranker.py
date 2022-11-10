import random
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from amazon_product_search.es.response import Result
from amazon_product_search.nlp.encoder import JA_SBERT


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        pass


class NoOpReranker(Reranker):
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        return results


class RandomReranker(Reranker):
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        return random.sample(results, len(results))


class SentenceBERTReranker(Reranker):
    def __init__(self, model_name: str = JA_SBERT, batch_size: int = 8):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cls_pooling(self, model_output: BaseModelOutput) -> Tensor:
        return model_output.last_hidden_state[:, 0]

    def encode(self, texts: list[str]):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.cls_pooling(model_output)
        return embeddings

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not query or not results:
            return results

        self.model.eval()

        with torch.no_grad():
            query_emb = self.encode([query]).repeat(len(results), 1)
            product_emb = self.encode([result.product["product_title"] for result in results])
            scores = torch.diagonal(torch.mm(query_emb, product_emb.transpose(0, 1)).to("cpu"))
        results = [result for result, score in sorted(zip(results, scores), key=lambda e: e[1], reverse=True)]
        return results


def to_string(reranker: Reranker) -> str:
    return reranker.__class__.__name__


def from_string(reranker_str: str) -> Reranker:
    return {
        "NoOpReranker": NoOpReranker,
        "RandomReranker": RandomReranker,
        "SentenceBERTReranker": SentenceBERTReranker,
    }[reranker_str]()
