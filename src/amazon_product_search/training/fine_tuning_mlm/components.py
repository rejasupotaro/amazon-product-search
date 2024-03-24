from typing import Any, Optional, Union

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from transformers import (
    AutoModelForMaskedLM,
)

from amazon_product_search.training.fine_tuning_mlm.data_module import ProductMLMDataModule
from amazon_product_search.training.shared.metric_logger import MetricLoggerPL


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


def run(
    project_dir: str,
    input_filename: str,
    bert_model_name: str,
    learning_rate: float = 1e-4,
    mlm_probability: float = 0.1,
    batch_size: int = 32,
    num_sentences: Optional[int] = None,
    max_epochs: int = 1,
    devices: Union[list[int], str, int] = "auto",
) -> list[dict[str, Any]]:
    data_dir = f"{project_dir}/data"
    models_dir = f"{project_dir}/models"

    df = pd.read_parquet(f"{data_dir}/{input_filename}")

    fine_tuner = MLMFineTuner(bert_model_name, learning_rate)
    data_module = ProductMLMDataModule(bert_model_name, df, mlm_probability, batch_size, num_sentences)
    metric_logger = MetricLoggerPL()
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{models_dir}/checkpoints/{bert_model_name}",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="16-mixed",
        devices=devices,
        callbacks=[metric_logger, model_checkpoint],
    )
    trainer.fit(fine_tuner, data_module)

    output_dir = f"{models_dir}/fine_tuned/{bert_model_name}"
    model = fine_tuner.model
    tokenizer = data_module.tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return metric_logger.metrics
