from dataclasses import asdict
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search import source
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.es.response import Response
from amazon_product_search.metrics import compute_ndcg, compute_recall, compute_zero_hit_rate
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking import reranker
from demo.experimental_setup import EXPERIMENTS, ExperimentalSetup, Variant
from demo.page_config import set_page_config
from demo.utils import split_fields

es_client = EsClient()
query_builder = QueryBuilder()


@st.cache
def load_labels(experimental_setup: ExperimentalSetup) -> pd.DataFrame:
    df = source.load_labels(experimental_setup.locale)
    if experimental_setup.num_queries:
        queries = df.sample(frac=1)["query"].unique()[: experimental_setup.num_queries]
        df = df[df["query"].isin(queries)]
    return df


def count_docs(index_name: str) -> int:
    return es_client.count_docs(index_name)


def draw_variants(variants: list[Variant]):
    variants_df = pd.DataFrame([asdict(variant) for variant in variants])
    variants_df["reranker"] = variants_df["reranker"].apply(reranker.to_string)
    st.write(variants_df)


def search(index_name: str, query: str, variant: Variant) -> Response:
    es_query = None
    es_knn_query = None

    sparse_fields, dense_fields = split_fields(variant.fields)
    if sparse_fields:
        es_query = query_builder.build_multimatch_search_query(
            query=query, fields=sparse_fields, is_synonym_expansion_enabled=variant.enable_synonym_expansion
        )
    if dense_fields:
        # TODO: Should multiple vector fields be handled?
        es_knn_query = query_builder.build_knn_search_query(query, dense_fields[0], top_k=variant.top_k)

    return es_client.search(index_name=index_name, query=es_query, knn_query=es_knn_query, size=variant.top_k)


def compute_metrics(index_name: str, query: str, variant: Variant, labels_df: pd.DataFrame) -> dict[str, Any]:
    response = search(index_name, query, variant)
    response.results = variant.reranker.rerank(query, response.results)

    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = set(labels_df[labels_df["esci_label"] == "E"]["product_id"].tolist())
    judgements: dict[str, str] = {row["product_id"]: row["esci_label"] for row in labels_df.to_dict("records")}
    return {
        "variant": variant.name,
        "query": query,
        "total_hits": response.total_hits,
        "recall": compute_recall(retrieved_ids, relevant_ids),
        "ndcg": compute_ndcg(retrieved_ids, judgements),
        "ndcg_prime": compute_ndcg(retrieved_ids, judgements, prime=True),
    }


def perform_search(experimental_setup: ExperimentalSetup, query_dict: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    total_examples = len(query_dict)
    n = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    metrics = []
    for query, query_labels_df in query_dict.items():
        n += 1
        query = normalize_query(query)
        progress_text.text(f"Query ({n} / {total_examples}): {query}")
        progress_bar.progress(n / total_examples)
        for variant in experimental_setup.variants:
            metrics.append(compute_metrics(experimental_setup.index_name, query, variant, query_labels_df))
    progress_text.text(f"Done ({n} / {total_examples})")
    return metrics


def compute_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = (
        metrics_df.groupby("variant")
        .agg(
            total_hits=("total_hits", lambda series: int(series.mean())),
            zero_hit_rate=("total_hits", lambda series: compute_zero_hit_rate(series.values)),
            recall=("recall", lambda series: series.mean().round(4)),
            ndcg=("ndcg", lambda series: series.mean().round(4)),
            ndcg_prime=("ndcg_prime", lambda series: series.mean().round(4)),
        )
        .reset_index()
    )
    return stats_df


def draw_figures(metrics_df: pd.DataFrame):
    for metric in ["total_hits", "recall", "ndcg"]:
        fig = px.box(metrics_df, y=metric, color="variant")
        st.plotly_chart(fig)


def main():
    set_page_config()
    st.write("## Experiments")

    experiment_name = st.selectbox("Experiment:", EXPERIMENTS.keys())
    experimental_setup = EXPERIMENTS[experiment_name]

    num_docs = count_docs(experimental_setup.index_name)

    labels_df = load_labels(experimental_setup)
    query_dict: dict[str, pd.DataFrame] = {}
    for query, query_labels_df in labels_df.groupby("query"):
        query_dict[query] = query_labels_df

    st.write("### Experimental Setup")
    content = f"""
    The experiment is conducted on `{experimental_setup.index_name}` containing `{num_docs}` docs in total.
    We send `{len(query_dict)}` queries to the index with different parameters shown below.
    Then, we compute Total Hits, Zero Hit Rate, Recall, and NDCG on each variant.
    """
    st.write(content)

    st.write("#### Variants")
    draw_variants(experimental_setup.variants)

    clicked = st.button("Run")

    if not clicked:
        return

    metrics = perform_search(experimental_setup, query_dict)

    st.write("----")

    st.write("### Experimental Results")
    metrics_df = pd.DataFrame(metrics)

    st.write("#### Metrics by query")
    with st.expander("click to expand"):
        st.write(metrics_df)

    st.write("#### Metrics by variant")
    stats_df = compute_stats(metrics_df)
    st.write(stats_df)

    st.write("#### Analysis")
    draw_figures(metrics_df)


if __name__ == "__main__":
    main()
