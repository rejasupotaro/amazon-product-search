from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search import source
from amazon_product_search.metrics import compute_ap, compute_ndcg, compute_recall, compute_zero_hit_rate
from amazon_product_search.models.search import RequestParams, Response, Result
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.sparse_retrieval import query_builder
from amazon_product_search.sparse_retrieval.es_client import EsClient

es_client = EsClient(
    es_host="http://localhost:9200",
)


@dataclass
class SparseSearchConfig:
    name: str
    use_description: bool = False
    use_bullet_point: bool = False
    use_brand: bool = False
    use_color_name: bool = False
    top_k: int = 100


@dataclass
class DenseSearchConfig:
    name: str
    top_k: int = 100


Variant = Union[SparseSearchConfig, DenseSearchConfig]


@st.cache
def load_labels(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def sparse_search(config: SparseSearchConfig, query: str) -> Response:
    params = RequestParams(
        query=query,
        use_description=config.use_description,
        use_bullet_point=config.use_bullet_point,
        use_brand=config.use_brand,
        use_color_name=config.use_color_name,
    )
    es_query = query_builder.build(params)
    es_response = es_client.search(index_name="products_jp", es_query=es_query, size=config.top_k)
    return Response(
        results=[Result(product=hit["_source"], score=hit["_score"]) for hit in es_response["hits"]["hits"]],
        total_hits=es_response["hits"]["total"]["value"],
    )


def dense_search(config: DenseSearchConfig, query: str) -> Response:
    return Response(
        results=[],
        total_hits=100,
    )


def compute_metrics(variant: Variant, query: str, labels_df: pd.DataFrame) -> dict[str, Any]:
    if isinstance(variant, SparseSearchConfig):
        response = sparse_search(variant, query)
    elif isinstance(variant, DenseSearchConfig):
        response = dense_search(variant, query)
    else:
        raise ValueError

    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = set(labels_df[labels_df["esci_label"] == "exact"]["product_id"].tolist())
    judgements: dict[str, str] = {row["product_id"]: row["esci_label"] for row in labels_df.to_dict("records")}
    return {
        "variant": variant.name,
        "query": query,
        "total_hits": response.total_hits,
        "recall": compute_recall(retrieved_ids, relevant_ids),
        "ap": compute_ap(retrieved_ids, relevant_ids),
        "ndcg": compute_ndcg(retrieved_ids, judgements),
    }


def compute_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = (
        metrics_df.groupby("variant")
        .agg(
            total_hits=("total_hits", lambda series: int(np.mean(series))),
            zero_hit_rate=("total_hits", lambda series: compute_zero_hit_rate(series.values)),
            recall=("recall", "mean"),
            map=("ap", "mean"),
            ndcg=("ndcg", "mean"),
        )
        .reset_index()
    )
    return stats_df


def draw_figures(metrics_df: pd.DataFrame):
    for metric in ["total_hits", "recall", "ap", "ndcg"]:
        fig = px.box(metrics_df, y=metric, color="variant")
        st.plotly_chart(fig)


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")

    st.write("## Offline Experiment")

    locale = "jp"
    nrows = 100

    variants = [
        # SparseSearchConfig(name="title", top_k=100),
        # SparseSearchConfig(name="title_description", use_description=True, top_k=100),
        # SparseSearchConfig(name="title_bullet_point", use_bullet_point=True, top_k=100),
        # SparseSearchConfig(name="title_brand", use_brand=True, top_k=100),
        # SparseSearchConfig(name="title_color_name", use_color_name=True, top_k=100),
        DenseSearchConfig(name="dense", top_k=100)
    ]
    st.write("#### Variants")
    st.write(variants)

    clicked = st.button("Run")

    if not clicked:
        return

    labels_df = load_labels(locale, nrows)
    query_dict: dict[str, pd.DataFrame] = {}
    for query, query_labels_df in labels_df.groupby("query"):
        query_dict[query] = query_labels_df

    total_examples = len(variants) * len(query_dict)
    n = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    metrics = []
    for variant in variants:
        for query, query_labels_df in query_dict.items():
            query = normalize_query(query)
            progress_text.text(f"Query ({n} / {total_examples}): {query}")
            n += 1
            progress_bar.progress(n / total_examples)
            metrics.append(compute_metrics(variant, query, query_labels_df))
    progress_text.text(f"Done ({n} / {total_examples})")

    st.write("#### Result")
    metrics_df = pd.DataFrame(metrics)
    stats_df = compute_stats(metrics_df)
    st.write(stats_df)
    draw_figures(metrics_df)


if __name__ == "__main__":
    main()
