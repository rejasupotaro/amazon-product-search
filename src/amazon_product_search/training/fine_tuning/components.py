import math
from typing import Any, Optional, Union

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
)

from amazon_product_search.training.shared.metric_logger import MetricLogger


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
        prepend_tag: bool = False,
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
        self.prepend_tag = prepend_tag

    @staticmethod
    def make_sentences(df: pd.DataFrame, prepend_tag: bool) -> list[str]:
        columns = [
            ("product_title", "title"),
            ("product_brand", "brand"),
            ("product_color", "color"),
            ("product_bullet_point", "bullet"),
            ("product_description", "desc"),
        ]
        df = df[[column[0] for column in columns]]
        df = df.drop_duplicates()

        sentences = []
        for row in df.to_dict("records"):
            if prepend_tag:
                sentence = " ".join([f"{tag}: {row[field_name]}" for field_name, tag in columns if row[field_name]])
            else:
                sentence = " ".join([row[field_name] for field_name, tag in columns if row[field_name]])
            sentences.append(sentence)
        return sentences

    def train_dataloader(self) -> DataLoader:
        sentences = self.make_sentences(self.train_df, self.prepend_tag)
        dataset = TokenizedSentencesDataset(sentences, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator)

    def val_dataloader(self) -> DataLoader:
        sentences = self.make_sentences(self.val_df, self.prepend_tag)
        dataset = TokenizedSentencesDataset(sentences, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)


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
