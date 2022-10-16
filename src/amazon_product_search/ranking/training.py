from typing import Dict, List

import pandas as pd
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from amazon_product_search.metrics import LABEL_TO_GAIN
from amazon_product_search.nlp.normalizer import normalize_doc

DATA_DIR = "data"
MODELS_DIR = "models"


def preprocess(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(normalize_doc)
    return df


def load_dataset(locale: str, num_examples: int) -> pd.DataFrame:
    queries_filepath = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
    queries_df = pd.read_csv(queries_filepath, nrows=num_examples)
    queries_df.fillna("", inplace=True)
    print(queries_df.columns)

    products_filepath = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    products_df = pd.read_csv(products_filepath, usecols=["product_id", "product_title", "product_brand"])
    products_df.fillna("", inplace=True)

    df = queries_df.merge(products_df, on="product_id", how="left")

    df = preprocess(df, ["query", "product_title", "product_brand"])
    df["product"] = df["product_title"] + " " + df["product_brand"]

    df["gain"] = df["esci_label"].apply(lambda label: LABEL_TO_GAIN[label])

    return df[["query", "product", "gain"]]


def train(locale: str, base_model_name: str, output_model_name: str, num_examples: int, test_size: float):
    print("1. Load data")
    df = load_dataset(locale, num_examples=num_examples)

    print("2. Prepare data loaders")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_examples = []
    for row in train_df.to_dict("records"):
        train_examples.append(InputExample(texts=[row["query"], row["product"]], label=float(row["gain"])))
    train_dataloader: DataLoader = DataLoader(train_examples, shuffle=True, batch_size=4, drop_last=True)

    test_examples = {}
    query_to_id: Dict[str, str] = {}
    for row in test_df.to_dict("records"):
        qid = query_to_id[row["query"]]
        if qid not in test_examples:
            test_examples[qid] = {"query": row["query"], "positive": set(), "negative": set()}
        if row["gain"] > 0:
            test_examples[qid]["positive"].add(row["product"])
        else:
            test_examples[qid]["negative"].add(row["product"])
    evaluator = CERerankingEvaluator(test_examples, name="train-eval")

    loss_fct = nn.MSELoss()
    # loss_fct = losses.OnlineContrastiveLoss(model=model.model)
    # loss_fct = losses.BatchAllTripletLoss(model=model.model)

    evaluation_steps = 5000
    warmup_steps = 5000
    lr = 7e-6

    print("3. Prepare Cross-encoder model")
    model = CrossEncoder(
        base_model_name,
        num_labels=1,
        max_length=512,
        default_activation_function=torch.nn.Identity(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print("4. Train Cross-encoder model")
    model_filepath = f"{MODELS_DIR}/{locale}/{output_model_name}"
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        epochs=1,
        warmup_steps=warmup_steps,
        output_path=f"{model_filepath}_tmp",
        optimizer_params={"lr": lr},
    )
    model.save(model_filepath)


if __name__ == "__main__":
    # For English
    # base_model_name = "cross-encoder/ms-marco-electra-base"
    # base_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # base_model_name = "sentence-transformers/msmarco-roberta-base-v3"

    # For Spanish
    # base_model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    # base_model_name = "bertin-project/bertin-roberta-base-spanish"

    # For Japanese
    base_model_name = "cl-tohoku/bert-base-japanese-v2"
    # base_model_name = "cl-tohoku/bert-base-japanese-char-v2"
    # base_model_name = "nlp-waseda/roberta-large-japanese"

    # Multi-lingual
    # baseline = "paraphrase-multilingual-mpnet-base-v2"
    # baseline = "stsb-xlm-r-multilingual"

    output_model_name = base_model_name.split()[-1]

    train(
        locale="jp",
        base_model_name=base_model_name,
        output_model_name=output_model_name,
        num_examples=100,
        test_size=0.2,
    )
