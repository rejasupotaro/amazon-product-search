import random
from os import path
from typing import Any, Literal, Optional

import polars as pl
import streamlit as st
from more_itertools import chunked

from amazon_product_search.constants import HF
from amazon_product_search.core import metrics, source
from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.reranking.reranker import (
    ColBERTReranker,
    DotReranker,
    NoOpReranker,
    RandomReranker,
    Reranker,
    SpladeReranker,
)
from amazon_product_search.core.retrieval.response import Result
from demo.page_config import set_page_config

RERANKER_NAMES = [
    "Random Reranker",
    "Dot Reranker",
    "ColBERT Reranker",
    "SPLADE Reranker",
]


def init_rerankers() -> dict[str, Reranker]:
    rerankers: dict[str, Reranker] = {}
    rerankers["Elasticsearch Results"] = NoOpReranker()
    rerankers["Random Reranker"] = RandomReranker()
    rerankers["Dot Reranker"] = DotReranker()
    if path.exists(HF.JP_COLBERT):
        rerankers["ColBERT Reranker"] = ColBERTReranker(model_filepath=HF.JP_COLBERT)
    if path.exists(HF.JP_SPLADE):
        rerankers["SPLADE Reranker"] = SpladeReranker(model_filepath=HF.JP_SPLADE)
    return rerankers


es_client = EsClient()
rerankers = init_rerankers()


@st.cache_data
def load_dataset(split: Literal["-", "train", "test"]) -> pl.DataaFrame:
    merged_df = source.load_merged(locale="jp")
    df = merged_df if split == "-" else merged_df.filter(pl.col("split") == split)
    return df


def search(query: str, doc_ids: list[str], id_to_label: dict[str, str]) -> list[Result]:
    doc_id_filter_clauses = [{"term": {"product_id": doc_id}} for doc_id in doc_ids]
    es_query = {
        "bool": {
            "filter": [
                {
                    "bool": {
                        "should": doc_id_filter_clauses,
                    }
                }
            ],
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["product_title", "product_bullet_point"],
                        "operator": "or",
                    }
                }
            ],
        }
    }
    response = es_client.search(index_name="products_jp", query=es_query)
    for result in response.results:
        result.product["query"] = query
        result.product["esci_label"] = id_to_label[result.product["product_id"]]
    return response.results


def compute_ndcg(products: list[dict[str, Any]]) -> Optional[float]:
    retrieved_ids = [product["product_id"] for product in products]
    id_to_label: dict[str, str] = {product["product_id"]: product["esci_label"] for product in products}
    return metrics.compute_ndcg(retrieved_ids, id_to_label)


def draw_results(results: list[Result]) -> None:
    products = [result.product for result in results]

    ndcg = compute_ndcg(products)
    st.write(f"NDCG: {ndcg}")

    df = pl.from_dicts(products)
    st.write(df.select(["esci_label", "query", "product_title", "product_id"]).to_pandas())


def draw_examples(query: str, results: list[Result]) -> None:
    for reranker_names in chunked(rerankers, 2):
        columns = st.columns(2)
        for i, reranker_name in enumerate(reranker_names):
            with columns[i]:
                st.write(f"#### {reranker_name}")
                random_results = rerankers[reranker_name].rerank(query, results)
                draw_results(random_results)


def run_comparison(
    df: pl.DataFrame,
    id_to_label: dict[str, str],
    num_queries: int,
    reranker_names: list[str],
) -> None:
    queries = df.get_column("query").unique().to_list()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    rows = []
    for i, query in enumerate(random.sample(queries, num_queries), start=1):
        progress_text.text(f"Query ({i} / {num_queries}): {query}")
        progress_bar.progress(i / num_queries)

        products = df.filter(pl.col("query") == query).to_dicts()
        results = [Result(product=product, score=1) for product in products]

        rows.append(
            {
                "query": query,
                "variant": "Elassticsearch",
                "ndcg": compute_ndcg(
                    [
                        result.product
                        for result in search(
                            query,
                            doc_ids=[result.product["product_id"] for result in results],
                            id_to_label=id_to_label,
                        )
                    ]
                ),
            }
        )
        for reranker_name in reranker_names:
            rows.append(
                {
                    "query": query,
                    "variant": reranker_name,
                    "ndcg": compute_ndcg(
                        [result.product for result in rerankers[reranker_name].rerank(query, results)]
                    ),
                }
            )

    metrics_df = pl.from_dicts(rows)
    stats_df = metrics_df.groupby("variant").agg([pl.col("ndcg").mean().round(4)])

    st.write("### Overall")
    st.write(stats_df.to_pandas())

    st.write("### Metrics by Query")
    st.write(metrics_df.to_pandas())


def main() -> None:
    set_page_config()
    st.write("## Reranking")

    split = st.selectbox("split", ["-", "train", "test"], index=2)
    df = load_dataset(split)
    id_to_label = {row["product_id"]: row["esci_label"] for row in df.select(["product_id", "esci_label"]).to_dicts()}

    st.write("### Example")

    queries = df["query"].unique()
    query = str(st.selectbox("Query:", queries))
    filtered_df = df.filter(pl.col("query") == query)

    products = filtered_df.to_dicts()
    results = [Result(product=product, score=1) for product in products]

    draw_examples(query, results)

    st.write("### Comparison")
    with st.form("experiment_setup"):
        num_queries = int(st.number_input("Num Queries:", value=100, min_value=1, max_value=1000))
        reranker_names = st.multiselect("Rerankers:", RERANKER_NAMES, default=RERANKER_NAMES)
        if not st.form_submit_button("Run"):
            return
    run_comparison(df, id_to_label, num_queries, reranker_names)


if __name__ == "__main__":
    main()
