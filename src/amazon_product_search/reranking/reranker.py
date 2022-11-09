import random
from abc import ABC, abstractmethod

from amazon_product_search.es.response import Result


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


def to_string(reranker: Reranker) -> str:
    return reranker.__class__.__name__


def from_string(reranker_str: str) -> Reranker:
    return {
        "NoOpReranker": NoOpReranker,
        "RandomReranker": RandomReranker,
    }[reranker_str]()
