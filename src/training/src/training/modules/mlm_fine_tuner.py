
import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from transformers import (
    AutoModelForMaskedLM,
)


class MLMFineTuner(LightningModule):
    def __init__(self, bert_model_name: str, learning_rate: float):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(bert_model_name)
        self.learning_rate = learning_rate

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self.model(input_ids).logits
        loss = cross_entropy(outputs.view(-1, self.model.config.vocab_size), labels.view(-1))
        return loss

    def training_step(self, batch: list[str], batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[str], batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
