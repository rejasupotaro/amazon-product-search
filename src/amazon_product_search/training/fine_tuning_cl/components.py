from typing import Any

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from amazon_product_search.training.shared.metric_logger import MetricLoggerST


def product_to_text(row: dict[str, Any], fields: list[str]) -> str:
    texts = []
    for field in fields:
        if row[field]:
            texts.append(row[field])
    return " ".join(texts)


class TripletDataset(Dataset):
    def __init__(self, examples: list[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def new_train_dataloader(df: pd.DataFrame, fields: list[str]) -> DataLoader:
    train_examples: list[InputExample] = []
    for i in range(len(df)):
        row = df.iloc[i]
        query = row["query"]
        positive = product_to_text(row, fields)
        negative = product_to_text(df.sample().to_dict("records")[0], fields)
        train_examples.append(
            InputExample(
                texts=[query, positive, negative],
            )
        )
    return DataLoader(TripletDataset(train_examples), shuffle=True, batch_size=16)


def new_val_examples(df: pd.DataFrame, fields: list[str]) -> list[InputExample]:
    val_examples = []
    for i in range(len(df)):
        row = df.iloc[i]
        query = row["query"]
        positive = product_to_text(row, fields)
        val_examples.append(InputExample(texts=[query, positive], label=1))
        negative = product_to_text(df.sample().to_dict("records")[0], fields)
        val_examples.append(InputExample(texts=[query, negative], label=0))
    return val_examples


def run(
    project_dir: str,
    input_filename: str,
    bert_model_name: str,
    max_epochs: int = 1,
) -> list[dict[str, Any]]:
    data_dir = f"{project_dir}/data"
    models_dir = f"{project_dir}/models"
    model_filepath = f"{models_dir}/fine_tuned/{bert_model_name}"

    df = pd.read_parquet(f"{data_dir}/{input_filename}")
    df = df[df["split"] == "train"]
    df = df.head(1000)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.head(100)
    val_df = val_df.head(100)

    fields = ["product_title"]
    dataloader = new_train_dataloader(train_df, fields)
    val_examples = new_val_examples(val_df, fields)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples=val_examples,
        main_similarity=SimilarityFunction.COSINE,
    )
    metric_logger = MetricLoggerST("val_spearman_cosine")

    model = SentenceTransformer(bert_model_name)
    loss = losses.TripletLoss(model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        evaluator=evaluator,
        epochs=max_epochs,
        warmup_steps=100,
        callback=metric_logger,
    )

    model.save(model_filepath)

    return metric_logger.metrics
