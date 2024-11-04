from typing import Literal

from torch import Tensor
from torch.nn import Module

PoolingMode = Literal["cls", "mean", "max"]


class Pooler(Module):
    def __init__(self, pooling_mode: PoolingMode):
        super().__init__()
        self.pooling_mode = pooling_mode

    def forward(self, token_embs: Tensor, attention_mask: Tensor):
        match self.pooling_mode:
            case "cls":
                text_emb = token_embs[:, 0]
            case "mean":
                attention_mask = attention_mask.unsqueeze(dim=-1)
                token_embs = token_embs * attention_mask
                text_emb = token_embs.sum(dim=1) / attention_mask.sum(dim=1)
            case "max":
                text_emb, _ = (token_embs * attention_mask.unsqueeze(dim=-1)).max(dim=1)
            case _:
                raise ValueError(
                    f"Unexpected pooling_mode is given: {self.pooling_mode}"
                )
        return text_emb
