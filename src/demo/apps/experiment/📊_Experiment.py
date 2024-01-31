import asyncio
import logging
import traceback
from dataclasses import asdict
from typing import Any

import plotly.express as px
import polars as pl
import streamlit as st

from amazon_product_search.constants import HF
from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.metrics import (
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_zero_hit_rate,
)
from amazon_product_search.core.parallel import limit_concurrency
from amazon_product_search.core.reranking import reranker
from amazon_product_search.core.retrieval.query_vector_cache import QueryVectorCache
from amazon_product_search.core.retrieval.response import Response
from amazon_product_search.core.retrieval.retriever import Retriever
from amazon_product_search.core.source import Locale
from demo import utils
from demo.apps.experiment.experiment_setup import EXPERIMENTS, ExperimentSetup, Variant
from demo.page_config import set_page_config

es_client = EsClient()


@st.cache_resource
def get_retriever(locale: Locale) -> Retriever:
    hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]
    vector_cache = QueryVectorCache()
    vector_cache.load(locale)
    query_builder = QueryBuilder(locale=locale, hf_model_name=hf_model_name, vector_cache=vector_cache)
    return Retriever(locale=locale, es_client=es_client, query_builder=query_builder)


@st.cache_data
def _load_labels(locale: str) -> pl.DataFrame:
    df = utils.load_labels(locale)
    df = df.filter(pl.col("split") == "test")
    return df


def load_labels(locale: Locale, num_queries: int) -> pl.DataFrame:
    df = _load_labels(locale)
    if num_queries:
        queries = df.get_column("query").sample(frac=1).unique()[:num_queries]
        df = df.filter(pl.col("query").is_in(queries))
    return df


def count_docs(index_name: str) -> int:
    return es_client.count_docs(index_name)


def draw_variants(variants: list[Variant]) -> None:
    rows = []
    for variant in variants:
        row = asdict(variant)
        row["reranker"] = reranker.to_string(row["reranker"])
        rows.append(row)
    variants_df = pl.from_dicts(rows)
    st.write(variants_df.to_pandas())


def search(locale: Locale, index_name: str, query: str, variant: Variant, labeled_ids: list[str] | None) -> Response:
    return get_retriever(locale).search(
        index_name=index_name,
        query=query,
        fields=variant.fields,
        query_type=variant.query_type,
        is_synonym_expansion_enabled=variant.enable_synonym_expansion,
        product_ids=labeled_ids,
        sparse_boost=variant.sparse_boost,
        dense_boost=variant.dense_boost,
        size=variant.top_k,
        rank_fusion=variant.rank_fusion,
    )


def compute_metrics(
    locale: Locale,
    index_name: str,
    experiment_setup: ExperimentSetup,
    variant: Variant,
    query: str,
    labels_df: pl.DataFrame,
) -> dict[str, Any]:
    labeled_ids = None
    if experiment_setup.task == "reranking":
        labeled_ids = labels_df.get_column("product_id").to_list()
    response = search(locale, index_name, query, variant, labeled_ids)
    response.results = variant.reranker.rerank(query, response.results)

    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = set(labels_df.filter(pl.col("esci_label") == "E").get_column("product_id").to_list())
    id_to_label: dict[str, str] = {row["product_id"]: row["esci_label"] for row in labels_df.to_dicts()}
    metric_dict = {
        "variant": variant.name,
        "query": query,
        "total_hits": response.total_hits,
        "R@10": compute_recall(retrieved_ids, relevant_ids, k=10),
        "R@100": compute_recall(retrieved_ids, relevant_ids, k=100),
        "NDCG@10": compute_ndcg(retrieved_ids, id_to_label, k=10),
        "NDCG@100": compute_ndcg(retrieved_ids, id_to_label, k=100),
    }
    if experiment_setup.task == "retrieval":
        precision_at_10 = compute_precision(retrieved_ids, relevant_ids, k=10)
        metric_dict["P@10"] = precision_at_10 if precision_at_10 is not None else 0
        precision_at_100 = compute_precision(retrieved_ids, relevant_ids, k=100)
        metric_dict["P@100"] = precision_at_100 if precision_at_100 is not None else 0
    return metric_dict


async def compute_metrics_by_variant(
    locale: Locale,
    index_name: str,
    experiment_setup: ExperimentSetup,
    query: str,
    query_labels_df: pl.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    metrics = []
    try:
        for variant in experiment_setup.variants:
            ms = compute_metrics(locale, index_name, experiment_setup, variant, query, query_labels_df)
            metrics.append(ms)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    await asyncio.sleep(0.1)
    return query, metrics


async def perform_search(
    locale: Locale, index_name: str, experiment_setup: ExperimentSetup, query_dict: dict[str, pl.DataFrame]
) -> list[dict[str, Any]]:
    coroutines = []
    for query, query_labels_df in query_dict.items():
        coroutine = compute_metrics_by_variant(locale, index_name, experiment_setup, query, query_labels_df)
        coroutines.append(coroutine)

    n, total_examples = 0, len(query_dict)
    progress_bar, progress_text = st.progress(0), st.empty()
    metrics = []
    for task in asyncio.as_completed(limit_concurrency(coroutines, max_concurrency=10)):
        query, ms = await task
        n += 1
        progress_bar.progress(n / total_examples)
        progress_text.text(f"{n} / {total_examples} | query: {query}")
        if ms:
            metrics.extend(ms)
    return metrics


def compute_stats(experiment_setup: ExperimentSetup, metrics_df: pl.DataFrame) -> pl.DataFrame:
    stats_df = metrics_df.groupby("variant").agg(
        [
            pl.col("total_hits").mean().cast(int),
            pl.col("total_hits").apply(lambda series: compute_zero_hit_rate(series.to_list())).alias("zero_hit_rate"),
        ]
        + (
            [
                pl.col("P@10").mean().round(4),
                pl.col("P@100").mean().round(4),
            ]
            if experiment_setup.task == "retrieval"
            else []
        )
        + [
            pl.col("R@10").mean().round(4),
            pl.col("R@100").mean().round(4),
            pl.col("NDCG@10").mean().round(4),
            pl.col("NDCG@100").mean().round(4),
        ]
    )
    return stats_df


def draw_figures(metrics_df: pl.DataFrame) -> None:
    for metric in ["P", "R", "NDCG"]:
        for column, k in zip(st.columns(2), [10, 100], strict=True):
            if f"{metric}@{k}" not in metrics_df.columns:
                continue
            with column:
                fig = px.box(metrics_df.to_pandas(), y=f"{metric}@{k}", color="variant")
                st.plotly_chart(fig)


async def f(count: int):
    container = st.empty()
    for i in range(11):
        v = count + i
        container.write(f"{v}")
        await asyncio.sleep(1)
    return v


async def runner():
    tasks = []
    for i in range(4):
        t = asyncio.create_task(f(10**i))
        tasks.append(t)
        await asyncio.sleep(1)
    values = await asyncio.gather(*tasks)
    return values


async def main() -> None:
    set_page_config()
    st.write("## Experiments")

    with st.sidebar:
        locale = st.selectbox("Locale:", ["us", "jp"])
        index_name = st.selectbox("Index:", es_client.list_indices())
        num_queries = st.number_input("Number of queries:", min_value=1, max_value=1000000, value=10)

    experiment_name = st.selectbox("Experiment:", EXPERIMENTS.keys())
    experiment_setup = EXPERIMENTS[experiment_name]

    num_docs = count_docs(index_name)

    labels_df = load_labels(locale, num_queries)
    query_dict = {}
    for query, query_labels_df in labels_df.groupby("query"):
        query_dict[str(query)] = query_labels_df

    st.write("### Experiment Setup")
    content = f"""
    The experiment is conducted on `{index_name}` containing `{num_docs}` docs in total.
    We send `{len(query_dict)}` queries to the index with different parameters shown below.
    Then, we compute Total Hits, Zero Hit Rate, Precision, Recall, and NDCG for each variant.
    """
    st.write(content)

    st.write("#### Variants")
    draw_variants(experiment_setup.variants)

    clicked = st.button("Run")

    if not clicked:
        return

    metrics = await perform_search(locale, index_name, experiment_setup, query_dict)

    st.write("----")

    st.write("### Experiment Results")
    metrics_df = pl.from_dicts(metrics)

    st.write("#### Metrics by query")
    with st.expander("click to expand"):
        st.write(metrics_df.to_pandas())

    st.write("#### Metrics by variant")
    stats_df = compute_stats(experiment_setup, metrics_df).to_pandas()
    stats_df = stats_df.sort_values("NDCG@100", ascending=False)
    st.write(stats_df)
    with st.expander("Metrics in markdown"):
        st.text(stats_df.to_markdown(index=False))

    st.write("#### Analysis")
    draw_figures(metrics_df)


if __name__ == "__main__":
    asyncio.run(main())
