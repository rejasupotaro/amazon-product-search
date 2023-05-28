from dataclasses import asdict
from typing import Any

import plotly.express as px
import polars as pl
import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.es.response import Response
from amazon_product_search.metrics import (
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_zero_hit_rate,
)
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking import reranker
from demo import utils
from demo.apps.search.experimental_setup import EXPERIMENTS, ExperimentalSetup, Variant
from demo.page_config import set_page_config

es_client = EsClient()
query_builder = QueryBuilder()


@st.cache_data
def _load_labels(locale: str) -> pl.DataFrame:
    df = utils.load_labels(locale)
    df = df.filter(pl.col("split") == "test")
    return df


def load_labels(experimental_setup: ExperimentalSetup) -> pl.DataFrame:
    df = _load_labels(experimental_setup.locale)
    if experimental_setup.num_queries:
        queries = (
            df.get_column("query")
            .sample(frac=1)
            .unique()[: experimental_setup.num_queries]
        )
        df = df.filter(pl.col("query").is_in(queries))
    return df


def count_docs(index_name: str) -> int:
    return es_client.count_docs(index_name)


def draw_variants(variants: list[Variant]):
    rows = []
    for variant in variants:
        row = asdict(variant)
        row["reranker"] = reranker.to_string(row["reranker"])
        rows.append(row)
    variants_df = pl.from_dicts(rows)
    st.write(variants_df.to_pandas())


def search(
    index_name: str, query: str, variant: Variant, labeled_ids: list[str] | None
) -> Response:
    es_query = None
    es_knn_query = None

    sparse_fields, dense_fields = utils.split_fields(variant.fields)
    if sparse_fields:
        es_query = query_builder.build_sparse_search_query(
            query=query,
            fields=sparse_fields,
            query_type=variant.query_type,
            is_synonym_expansion_enabled=variant.enable_synonym_expansion,
            product_ids=labeled_ids,
        )
    if dense_fields:
        # TODO: Should multiple vector fields be handled?
        es_knn_query = query_builder.build_dense_search_query(
            query, dense_fields[0], boost=variant.dense_boost, top_k=variant.top_k
        )

    return es_client.search(
        index_name=index_name,
        query=es_query,
        knn_query=es_knn_query,
        size=variant.top_k,
    )


def compute_metrics(
    experimental_setup: ExperimentalSetup,
    variant: Variant,
    query: str,
    labels_df: pl.DataFrame,
) -> dict[str, Any]:
    labeled_ids = None
    if experimental_setup.task == "reranking":
        labeled_ids = labels_df.get_column("product_id").to_list()
    response = search(experimental_setup.index_name, query, variant, labeled_ids)
    response.results = variant.reranker.rerank(query, response.results)

    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = set(
        labels_df.filter(pl.col("esci_label") == "E").get_column("product_id").to_list()
    )
    judgements: dict[str, str] = {
        row["product_id"]: row["esci_label"] for row in labels_df.to_dicts()
    }
    metric_dict = {
        "variant": variant.name,
        "query": query,
        "total_hits": response.total_hits,
        "recall": compute_recall(retrieved_ids, relevant_ids),
        "ndcg": compute_ndcg(retrieved_ids, judgements),
        "ndcg_prime": compute_ndcg(retrieved_ids, judgements, prime=True),
    }
    if experimental_setup.task == "retrieval":
        precision = compute_precision(retrieved_ids, relevant_ids, k=10)
        metric_dict["precision@10"] = precision if precision is None else 0
    return metric_dict


def perform_search(
    experimental_setup: ExperimentalSetup, query_dict: dict[str, pl.DataFrame]
) -> list[dict[str, Any]]:
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
            metrics.append(
                compute_metrics(experimental_setup, variant, query, query_labels_df)
            )
    progress_text.text(f"Done ({n} / {total_examples})")
    return metrics


def compute_stats(
    experimental_setup: ExperimentalSetup, metrics_df: pl.DataFrame
) -> pl.DataFrame:
    stats_df = metrics_df.groupby("variant").agg(
        [
            pl.col("total_hits").mean().cast(int),
            pl.col("total_hits")
            .apply(lambda series: compute_zero_hit_rate(series.to_list()))
            .alias("zero_hit_rate"),
        ]
        + (
            [pl.col("precision@10").mean().round(4)]
            if experimental_setup.task == "retrieval"
            else []
        )
        + [
            pl.col("recall").mean().round(4),
            pl.col("ndcg").mean().round(4),
            pl.col("ndcg_prime").mean().round(4),
        ]
    )
    return stats_df


def draw_figures(metrics_df: pl.DataFrame):
    for metric in ["total_hits", "recall", "ndcg"]:
        fig = px.box(metrics_df.to_pandas(), y=metric, color="variant")
        st.plotly_chart(fig)


def main():
    set_page_config()
    st.write("## Experiments")

    experiment_name = st.selectbox("Experiment:", EXPERIMENTS.keys())
    experimental_setup = EXPERIMENTS[experiment_name]

    num_docs = count_docs(experimental_setup.index_name)

    labels_df = load_labels(experimental_setup)
    query_dict: dict[str, pl.DataFrame] = {}
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
    metrics_df = pl.from_dicts(metrics)

    st.write("#### Metrics by query")
    with st.expander("click to expand"):
        st.write(metrics_df.to_pandas())

    st.write("#### Metrics by variant")
    stats_df = compute_stats(experimental_setup, metrics_df)
    st.write(stats_df.to_pandas())

    st.write("#### Analysis")
    draw_figures(metrics_df)


if __name__ == "__main__":
    main()
