import math
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForWholeWordMask,
)


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


class ProductMLMDataModule(LightningDataModule):
    def __init__(
        self,
        bert_model_name: str,
        df: pd.DataFrame,
        mlm_probability: float,
        batch_size: int,
        num_sentences: Optional[int] = None,
        prepend_tag: bool = False,
    ) -> None:
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
