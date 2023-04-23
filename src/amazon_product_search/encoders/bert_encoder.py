from typing import Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from amazon_product_search.constants import MODELS_DIR


class BertEncoder(nn.Module):
    def __init__(self, bert_model_name: str, num_hidden: int = 768, num_proj: int = 128, trainable: bool = True):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        self.projection = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_proj),
        )

    @staticmethod
    def from_state(bert_model_name: str, model_name: str, models_dir: str = MODELS_DIR) -> "BertEncoder":
        model_filepath = f"{models_dir}/{model_name}"
        encoder = BertEncoder(bert_model_name)
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

    def forward(self, tokens: Dict[str, Tensor]) -> Tensor:
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        vec = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        vec, _ = torch.max(vec * attention_mask.unsqueeze(-1), dim=1)
        vec = self.projection(vec)
        return torch.nn.functional.normalize(vec, p=2, dim=1)

    def encode(self, texts: str | List[str]) -> Tensor:
        tokens = self.tokenize(texts)
        return self.forward(tokens)
