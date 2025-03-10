from typing import Any

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from training.shared.metric_logger import MetricLoggerST


def query_to_text(query: str, with_tag: bool) -> str:
    if with_tag:
        return f"query: {query}"
    else:
        return query


def product_to_text(row: dict[str, Any], fields: list[str], with_tag: bool) -> str:
    texts = []
    for field in fields:
        if row[field]:
            if with_tag:
                texts.append(f"{field}: {row[field]}")
            else:
                texts.append(row[field])
    return " ".join(texts)


class TripletDataset(Dataset):
    def __init__(self, examples: list[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def new_train_dataloader(df: pd.DataFrame, fields: list[str], with_tag: bool) -> DataLoader:
    train_examples: list[InputExample] = []
    for i in range(len(df)):
        row = df.iloc[i]
        query = query_to_text(row["query"], with_tag)
        positive = product_to_text(row, fields, with_tag)
        negative = product_to_text(df.sample().to_dict("records")[0], fields, with_tag)
        train_examples.append(
            InputExample(
                texts=[query, positive, negative],
            )
        )
    return DataLoader(TripletDataset(train_examples), shuffle=True, batch_size=16)


def new_val_examples(df: pd.DataFrame, fields: list[str], with_tag: bool) -> list[InputExample]:
    val_examples = []
    for i in range(len(df)):
        row = df.iloc[i]
        query = query_to_text(row["query"], with_tag)

        positive = product_to_text(row, fields, with_tag)
        val_examples.append(InputExample(texts=[query, positive], label=1))

        negative = product_to_text(df.sample().to_dict("records")[0], fields, with_tag)
        val_examples.append(InputExample(texts=[query, negative], label=0))
    return val_examples


def run(
    project_dir: str,
    input_filename: str,
    bert_model_name: str,
    max_epochs: int = 1,
    debug: bool = True,
    with_tag: bool = False,
) -> list[dict[str, Any]]:
    data_dir = f"{project_dir}/data"
    models_dir = f"{project_dir}/models"
    model_filepath = f"{models_dir}/fine_tuned/{bert_model_name}"

    df = pd.read_parquet(f"{data_dir}/{input_filename}")
    df = df[df["split"] == "train"]
    if debug:
        df = df.head(1000)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(100)

    fields = ["product_title"]
    dataloader = new_train_dataloader(train_df, fields, with_tag)
    val_examples = new_val_examples(val_df, fields, with_tag)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples=val_examples,
        main_similarity=SimilarityFunction.COSINE,
    )
    metric_logger = MetricLoggerST("val_spearman_cosine")

    model = SentenceTransformer(bert_model_name)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        evaluator=evaluator,
        epochs=max_epochs,
        warmup_steps=100,
        callback=metric_logger,
    )

    model.save(model_filepath)

    return metric_logger.metrics
