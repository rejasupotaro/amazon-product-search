import itertools
import random

import pandas as pd
from lightning import LightningDataModule
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm

from dense_retrieval.encoders import BiEncoder
from dense_retrieval.encoders.text_encoder import Tokenizer
from dense_retrieval.retrievers.ann_index import ANNIndex


class AmazonDataset(Dataset):
    def __init__(self, examples: list[tuple[str, str, str]]):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return self.examples[index]


class AmazonDataLoader(DataLoader):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(collate_fn=self.collate_fn, **kwargs)

    def tokenize(self, texts: list[str]) -> dict[str, Tensor]:
        return self.tokenizer.tokenize(texts)

    def collate_fn(  # type: ignore
        self, batch: list[tuple[str, str, str]]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        query, pos_doc, neg_doc = zip(*batch, strict=True)
        query_tokens = self.tokenize(list(query))
        pos_doc_tokens = self.tokenize(list(pos_doc))
        neg_doc_tokens = self.tokenize(list(neg_doc))
        return query_tokens, pos_doc_tokens, neg_doc_tokens


class AmazonDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        bi_encoder: BiEncoder,
        filename: str = "merged_jp.parquet",
        train_datasize: int | None = None,
        test_datasize: int | None = None,
        batch_size: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.filename = filename
        self.bi_encoder = bi_encoder
        self.train_datasize = train_datasize
        self.test_datasize = test_datasize
        self.batch_size = batch_size

    def setup(self, stage: str):
        df = pd.read_parquet(f"{self.data_dir}/{self.filename}")
        self.train_df = df[df["split"] == "train"]
        self.test_df = df[df["split"] == "test"].sample(frac=1)

    def create_ann_dataset(self, df: pd.DataFrame) -> AmazonDataset:
        ann_index = ANNIndex(dim=768)
        product_ids = df["product_id"].tolist()
        product_titles = df["product_title"].tolist()
        it = list(chunked(zip(product_ids, product_titles, strict=True), 128))
        for batch in tqdm(it):
            doc_embs = self.bi_encoder.product_encoder([e[1] for e in batch])
            ann_index.add_items([e[0] for e in batch], doc_embs)
        ann_index.build()

        doc_id_to_title = {
            row["product_id"]: row["product_title"] for row in df.to_dict("records")
        }

        examples: list[tuple[str, str, str]] = []
        for query, group_df in df.groupby("query"):
            pos_doc_ids = group_df[group_df["esci_label"] == "E"]["product_id"].tolist()
            if not pos_doc_ids:
                continue
            neg_doc_ids = group_df[group_df["esci_label"] == "I"]["product_id"].tolist()

            encoded_query = self.bi_encoder.query_encoder(query)
            max_num_docs = 6
            similar_doc_ids, scores = ann_index.search(encoded_query, top_k=2)
            similar_doc_ids = [
                doc_id for doc_id in similar_doc_ids if doc_id not in pos_doc_ids
            ]
            num_neg_docs_to_sample = max_num_docs - len(similar_doc_ids)
            neg_doc_ids = random.choices(
                neg_doc_ids, k=min(len(neg_doc_ids), num_neg_docs_to_sample)
            )
            neg_doc_ids = neg_doc_ids + similar_doc_ids

            pos_doc_ids = random.choices(
                pos_doc_ids, k=min(len(pos_doc_ids), max_num_docs)
            )
            for pos_doc_id, neg_doc_id in itertools.product(pos_doc_ids, neg_doc_ids):
                examples.append(
                    (query, doc_id_to_title[pos_doc_id], doc_id_to_title[neg_doc_id])
                )
        return AmazonDataset(examples)

    def create_dataset(self, df: pd.DataFrame) -> AmazonDataset:
        examples: list[tuple[str, str, str]] = []
        for query, group_df in df.groupby("query"):
            pos_docs = group_df[group_df["esci_label"] == "E"]["product_title"].tolist()
            neg_docs = group_df[group_df["esci_label"] == "I"]["product_title"].tolist()
            for pos_doc, neg_doc in itertools.product(pos_docs, neg_docs):
                examples.append((query, pos_doc, neg_doc))
        return AmazonDataset(examples)

    def create_dataloader(
        self, dataset: AmazonDataset, shuffle: bool = False
    ) -> AmazonDataLoader:
        return AmazonDataLoader(
            tokenizer=self.bi_encoder.tokenizer,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> AmazonDataLoader:
        df = self.train_df
        if self.train_datasize:
            df = df.head(self.train_datasize)
        dataset = self.create_ann_dataset(df)
        return self.create_dataloader(dataset, shuffle=True)

    def val_dataloader(self) -> AmazonDataLoader:
        df = self.test_df
        if self.test_datasize:
            df = df.head(self.test_datasize)
        dataset = self.create_dataset(df)
        return self.create_dataloader(dataset)

    def test_dataloader(self) -> AmazonDataLoader:
        df = self.test_df
        if self.test_datasize:
            df = df.head(self.test_datasize)
        dataset = self.create_dataset(df)
        return self.create_dataloader(dataset)
