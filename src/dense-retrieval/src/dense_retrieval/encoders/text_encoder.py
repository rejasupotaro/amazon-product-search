from typing import Protocol

import numpy as np
import torch
from more_itertools import chunked
from torch import Tensor
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer

from dense_retrieval.encoders.modules.pooler import (
    Pooler,
    PoolingMode,
)


class Tokenizer(Protocol):
    def tokenize(self, texts: str | list[str]) -> dict[str, Tensor]:
        ...


def save(models_dir: str, model_name: str, encoder: Module) -> str:
    model_filepath = f"{models_dir}/{model_name}.pt"
    torch.save(encoder.state_dict(), model_filepath)
    return model_filepath


def load(
    models_dir: str,
    model_name: str,
    hf_model_name: str,
    hf_model_trainable: bool,
    pooling_mode: PoolingMode,
) -> Module:
    model_filepath = f"{models_dir}/{model_name}.pt"
    encoder = TextEncoder(
        hf_model_name=hf_model_name,
        hf_model_trainable=hf_model_trainable,
        pooling_mode=pooling_mode,
    )
    encoder.load_state_dict(torch.load(model_filepath))
    return encoder


class TextEncoder(Module):
    def __init__(
        self,
        hf_model_name: str,
        hf_model_trainable: bool,
        pooling_mode: PoolingMode,
    ) -> None:
        super().__init__()
        self.hf_model_name = hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.bert_model = AutoModel.from_pretrained(hf_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = hf_model_trainable
        self.pooler = Pooler(pooling_mode)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def tokenize(self, texts: str | list[str]) -> dict[str, Tensor]:
        tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokens

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        token_embs = self.bert_model(**tokens).last_hidden_state
        text_emb = self.pooler.forward(token_embs, tokens["attention_mask"])
        return text_emb

    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        input_was_string = False
        if isinstance(texts, str):
            input_was_string = True
            texts = texts

        self.eval()
        all_embs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in chunked(texts, n=batch_size):
                tokens = self.tokenize(batch)
                for key in tokens:
                    if isinstance(tokens[key], Tensor):
                        tokens[key] = tokens[key].to(self.device)
                embs: Tensor = self(tokens)
                embs = embs.detach().cpu().numpy()
                all_embs.extend(embs)
        return np.array(all_embs)[0] if input_was_string else np.array(all_embs)
