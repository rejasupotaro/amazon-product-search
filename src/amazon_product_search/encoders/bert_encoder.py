from typing import Dict, List, Literal

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from amazon_product_search.constants import MODELS_DIR


class BERTEncoder(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool = False,
        rep_mode: Literal["mean", "max", "cls"] = "mean",
        num_hidden: int = 768,
        num_proj: int = 128,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = bert_model_trainable
        self.rep_mode = rep_mode
        self.projection = nn.Sequential(
            nn.Linear(num_hidden, num_proj),
        )

    @staticmethod
    def from_state(bert_model_name: str, model_name: str, models_dir: str = MODELS_DIR) -> "BERTEncoder":
        model_filepath = f"{models_dir}/{model_name}"
        encoder = BERTEncoder(bert_model_name)
        encoder.load_state_dict(torch.load(model_filepath))
        return encoder

    def tokenize(self, texts: str | List[str]) -> Dict[str, Tensor]:
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def convert_to_single_vector(self, vec: Tensor, attention_mask: Tensor) -> Tensor:
        match self.rep_mode:
            case "mean":
                vec, _ = (vec * attention_mask.unsqueeze(-1)).max(dim=1)
            case "max":
                vec = (vec * attention_mask.unsqueeze(-1)).mean(dim=1)
            case "cls":
                vec = vec[:, 0]
            case _:
                raise ValueError(f"Unexpected rep_mode is given: {self.rep_mode}")
        return vec

    def forward(self, tokens: Dict[str, Tensor]) -> Tensor:
        vec = self.bert_model(**tokens).last_hidden_state
        vec = self.convert_to_single_vector(vec, tokens["attention_mask"])
        vec = self.projection(vec)
        return torch.nn.functional.normalize(vec, p=2, dim=1)

    def encode(self, texts: str | List[str]) -> Tensor:
        tokens = self.tokenize(texts)
        return self.forward(tokens)
