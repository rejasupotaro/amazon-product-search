import math
from typing import Any, Optional, Union

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
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


class TokenizedSentencesDataset(Dataset):
    def __init__(self, sentences: list[str], tokenizer: AutoTokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.tokenizer(
            self.sentences[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )


class MLMDataModule(LightningDataModule):
    def __init__(
        self,
        bert_model_name: str,
        df: pd.DataFrame,
        mlm_probability: float,
        batch_size: int,
        num_sentences: Optional[int] = None,
    ):
        super().__init__()
        self.train_df = df[df["split"] == "train"]
        self.val_df = df[df["split"] == "test"]
        if num_sentences:
            self.train_df = self.train_df.head(num_sentences)
            train_size = len(self.train_df)
            val_size = math.ceil(train_size * 0.2)
            self.val_df = self.val_df.head(val_size)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.collator = DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        sentences = self.train_df["product_title"].unique().tolist()
        dataset = TokenizedSentencesDataset(sentences, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator)

    def val_dataloader(self) -> DataLoader:
        sentences = self.val_df["product_title"].unique().tolist()
        dataset = TokenizedSentencesDataset(sentences, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)


class MetricLogger(Callback):
    def __init__(self):
        self.metrics = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics["train_loss"]
        self.metrics.append(
            {
                "epoch": epoch,
                "train_loss": loss.detach().cpu().numpy().round(4),
            }
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics["val_loss"]
        self.metrics.append(
            {
                "epoch": epoch,
                "val_loss": loss.detach().cpu().numpy().round(4),
            }
        )


def run(
    data_dir: str,
    input_filename: str,
    bert_model_name: str,
    learning_rate: float = 1e-4,
    mlm_probability: float = 0.1,
    batch_size: int = 32,
    num_sentences: Optional[int] = None,
    max_epochs: int = 1,
    devices: Union[list[int], str, int] = "auto",
) -> list[dict[str, Any]]:
    df = pd.read_parquet(f"{data_dir}/{input_filename}")

    fine_tuner = MLMFineTuner(bert_model_name, learning_rate)
    data_module = MLMDataModule(bert_model_name, df, mlm_probability, batch_size, num_sentences)
    metric_logger = MetricLogger()

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="16-mixed",
        devices=devices,
        callbacks=[metric_logger],
    )
    trainer.fit(fine_tuner, data_module)

    output_dir = f"{data_dir}/fine_tuned/{bert_model_name}"
    model = fine_tuner.model
    tokenizer = data_module.tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return metric_logger.metrics
