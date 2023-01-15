from typing import Any, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from amazon_product_search.constants import HF


class ColBERTer(nn.Module):
    def __init__(self, bert_model_name: str):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        trainable = False
        for p in self.bert_model.parameters():
            p.requires_grad = trainable
        self.score_merger = nn.Parameter(torch.zeros(1))
        self.compression_dim = 128
        self.compressor = nn.Linear(self.bert_model.config.hidden_size, self.compression_dim)
        self.stopword_reducer = nn.Linear(self.compression_dim, 1, bias=True)
        nn.init.constant_(self.stopword_reducer.bias, 1)

    def forward(self, query: dict[str, Tensor], doc: dict[str, Any]):
        query_cls_vec, query_vecs, query_mask = self.encode_query(query)
        doc_cls_vec, doc_vecs, doc_mask, token_importance = self.encode_doc(doc)

        cls_score = self.compute_cls_score(query_cls_vec, doc_cls_vec)
        term_score = self.compute_term_score(query_vecs, query_mask, doc_vecs, doc_mask, exact_scoring_mask=None)

        weight = torch.sigmoid(self.score_merger)
        cls_score *= weight
        term_score *= 1 - weight
        score = cls_score + term_score

        return score, cls_score, term_score

    def encode_query(self, query: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        cls_vec, token_vecs, token_mask = self.encode_shared(query)
        token_vecs = token_vecs * token_mask.unsqueeze(-1)
        return cls_vec, token_vecs, token_mask

    def encode_doc(self, doc: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cls_vec, token_vecs, token_mask = self.encode_shared(doc)
        token_importance = nn.functional.relu(self.stopword_reducer(token_vecs))
        token_vecs = token_vecs * token_importance
        token_vecs = token_vecs * token_mask.unsqueeze(-1)
        # token_mask[token_importance.squeeze(-1) <= 0] = False
        return cls_vec, token_vecs, token_mask, token_importance

    def encode_shared(self, tokens: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        token_mask = tokens["attention_mask"].bool()
        vecs = self.bert_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])[0]
        cls_vecs = vecs[:, 0, :]
        token_vecs = self.compressor(vecs)
        return cls_vecs, token_vecs, token_mask

    def compute_cls_score(self, query_cls_vec: Tensor, doc_cls_vec: Tensor) -> Tensor:
        cls_score = (query_cls_vec * doc_cls_vec).sum(dim=1)
        return cls_score

    def compute_term_score(
        self,
        query_vecs: Tensor,
        query_mask: Tensor,
        doc_vecs: Tensor,
        doc_mask: Tensor,
        exact_scoring_mask: Optional[Tensor] = None,
    ) -> Tensor:
        score_per_term = torch.bmm(query_vecs, doc_vecs.transpose(2, 1))

        if exact_scoring_mask is not None:
            score_per_term[~exact_scoring_mask] = 0

        score_per_term[~(doc_mask).unsqueeze(1).expand(-1, score_per_term.shape[1], -1)] = -1000
        term_score = score_per_term.max(-1).values
        term_score[~query_mask] = 0
        term_score = term_score.sum(-1)
        return term_score


class ColBERTWrapper:
    def __init__(self, model_filepath: str = HF.JP_COLBERT, bert_model_name: str = "cl-tohoku/bert-base-japanese-v2"):
        self.colberter = ColBERTer(bert_model_name)
        self.colberter.load_state_dict(torch.load(model_filepath))
        self.colberter.eval()
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
