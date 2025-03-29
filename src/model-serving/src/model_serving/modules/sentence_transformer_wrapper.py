import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer


class SentenceTransformerWrapper(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        # When exporting a model to ONNX, using CPU is generally the preferred and safe choice.
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")
        self.to("cpu")

    @property
    def transformer(self) -> torch.nn.Module:
        return self.model[0]

    @property
    def pooling(self) -> torch.nn.Module:
        return self.model[1]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self.model.tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        transformer_output = self.transformer(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        sentence_embeddings = self.pooling(transformer_output)
        return sentence_embeddings["sentence_embedding"]
