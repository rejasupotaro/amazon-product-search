import random
from os import path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from more_itertools import chunked

from amazon_product_search import metrics
from amazon_product_search.constants import HF
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.response import Result
from amazon_product_search.reranking.reranker import (
    ColBERTReranker,
    DotReranker,
    NoOpReranker,
    RandomReranker,
    Reranker,
    SpladeReranker,
)
from demo.page_config import set_page_config
from demo.utils import load_merged


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


@st.cache
def extract_judgements(df: pd.DataFrame) -> dict[str, str]:
    return {row["product_id"]: row["esci_label"] for row in df[["product_id", "esci_label"]].to_dict("records")}


def search(query: str, doc_ids: list[str], all_judgements: dict[str, str]) -> list[Result]:
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
        result.product["esci_label"] = all_judgements[result.product["product_id"]]
    return response.results


def compute_ndcg(products: list[dict[str, Any]]) -> Optional[float]:
    retrieved_ids = [product["product_id"] for product in products]
    judgements: dict[str, str] = {product["product_id"]: product["esci_label"] for product in products}
    ndcg = metrics.compute_ndcg(retrieved_ids, judgements)
    if ndcg:
        ndcg = round(ndcg, 4)
    return ndcg


def draw_results(results: list[Result]):
    products = [result.product for result in results]

    ndcg = compute_ndcg(products)
    st.write(f"NDCG: {ndcg}")

    df = pd.DataFrame(products)
    st.write(df[["esci_label", "query", "product_title", "product_id"]])


def draw_examples(query: str, results: list[Result]):
    for reranker_names in chunked(rerankers, 2):
        columns = st.columns(2)
        for i, reranker_name in enumerate(reranker_names):
            with columns[i]:
                st.write(f"#### {reranker_name}")
                random_results = rerankers[reranker_name].rerank(query, results)
                draw_results(random_results)


def run_comparison(df: pd.DataFrame, all_judgements: dict[str, str], num_queries: int):
    queries = list(df["query"].unique())
    n = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    rows = []
    for query in random.sample(queries, num_queries):
        n += 1
        progress_text.text(f"Query ({n} / {num_queries}): {query}")
        progress_bar.progress(n / num_queries)

        filtered_df = df[df["query"] == query]
        products = filtered_df.to_dict("records")
        results = [Result(product=product, score=1) for product in products]

        rows.append(
            {
                "query": query,
                "variant": "random",
                "ndcg": compute_ndcg(
                    [result.product for result in rerankers["Random Reranker"].rerank(query, results)]
                ),
            }
        )
        rows.append(
            {
                "query": query,
                "variant": "search",
                "ndcg": compute_ndcg(
                    [
                        result.product
                        for result in search(
                            query,
                            doc_ids=[result.product["product_id"] for result in results],
                            all_judgements=all_judgements,
                        )
                    ]
                ),
            }
        )
        rows.append(
            {
                "query": query,
                "variant": "dot",
                "ndcg": compute_ndcg([result.product for result in rerankers["Dot Reranker"].rerank(query, results)]),
            }
        )
        rows.append(
            {
                "query": query,
                "variant": "colbert",
                "ndcg": compute_ndcg(
                    [result.product for result in rerankers["ColBERT Reranker"].rerank(query, results)]
                ),
            }
        )
    metrics_df = pd.DataFrame(rows)
    stats_df = (
        metrics_df.groupby("variant")
        .agg(
            ndcg=("ndcg", lambda series: series.mean().round(4)),
        )
        .reset_index()
    )
    st.write(stats_df)


def main():
    set_page_config()
    st.write("## Reranking")

    merged_df = load_merged(locale="jp")
    split = st.selectbox("split", ["-", "train", "test"], index=2)
    if split == "-":
        df = merged_df
    else:
        df = merged_df[merged_df["split"] == split]

    all_judgements = extract_judgements(df)

    st.write("### Example")

    queries = df["query"].unique()
    query = st.selectbox("Query:", queries)
    filtered_df = df[df["query"] == query]

    products = filtered_df.to_dict("records")
    results = [Result(product=product, score=1) for product in products]

    draw_examples(query, results)

    st.write("### Comparison")
    if st.button("Run"):
        run_comparison(df, all_judgements, num_queries=10)


if __name__ == "__main__":
    main()
