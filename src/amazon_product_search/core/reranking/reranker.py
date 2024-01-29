import random
from typing import Protocol

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from amazon_product_search.constants import HF
from amazon_product_search.core.modules.colbert import ColBERTWrapper
from amazon_product_search.core.modules.splade import Splade
from amazon_product_search.core.retrieval.response import Result


class Reranker(Protocol):
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        ...


class NoOpReranker:
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        return results


class RandomReranker:
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        return random.sample(results, len(results))


class DotReranker:
    def __init__(self, model_name: str = HF.JP_SLUKE_MEAN, batch_size: int = 8) -> None:
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self, texts: list[str]) -> dict[str, Tensor]:
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            # max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def cls_pooling(self, model_output: BaseModelOutput) -> Tensor:
        return model_output.last_hidden_state[:, 0]

    def encode(self, texts: list[str]) -> Tensor:
        tokenized_texts = self.tokenize(texts)
        with torch.no_grad():
            tokens = self.model(**tokenized_texts, return_dict=True)
        cls_vecs = self.cls_pooling(tokens)
        return cls_vecs

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not query or not results:
            return results

        with torch.no_grad():
            query_cls_vec = self.encode([query]).repeat(len(results), 1)
            product_cls_vec = self.encode([result.product["product_title"] for result in results])
            scores = torch.diagonal(torch.mm(query_cls_vec, product_cls_vec.transpose(0, 1)))
        results = [
            result for result, score in sorted(zip(results, scores, strict=True), key=lambda e: e[1], reverse=True)
        ]
        return results


class ColBERTReranker(ColBERTWrapper):
    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not query or not results:
            return results

        with torch.no_grad():
            tokenized_queries = self.tokenize([query] * len(results))
            products = [result.product["product_title"] for result in results]
            tokenized_products = self.tokenize(products)
            scores, _, _ = self.colberter(tokenized_queries, tokenized_products)
            scores = scores.numpy()
        results = [
            result for result, score in sorted(zip(results, scores, strict=True), key=lambda e: e[1], reverse=True)
        ]
        return results


class SpladeReranker:
    def __init__(
        self,
        model_filepath: str = HF.JP_SPLADE,
        bert_model_name: str = "cl-tohoku/bert-base-japanese-v2",
    ):
        self.splade = Splade(bert_model_name)
        self.splade.load_state_dict(torch.load(model_filepath))
        self.splade.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def tokenize(self, texts: list[str]) -> dict[str, Tensor]:
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            # max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def rerank(self, query: str, results: list[Result]) -> list[Result]:
        if not query or not results:
            return results

        with torch.no_grad():
            tokenized_queries = self.tokenize([query] * len(results))
            products = [result.product["product_title"] for result in results]
            tokenized_products = self.tokenize(products)
            scores, _, _ = self.splade(tokenized_queries, tokenized_products)
            scores = scores.squeeze(-1).numpy()
        results = [
            result for result, score in sorted(zip(results, scores, strict=True), key=lambda e: e[1], reverse=True)
        ]
        return results


def to_string(reranker: Reranker) -> str:
    return reranker.__class__.__name__


def from_string(reranker_str: str) -> Reranker:
    return {
        "NoOpReranker": NoOpReranker,
        "RandomReranker": RandomReranker,
        "DotReranker": DotReranker,
    }[reranker_str]()
