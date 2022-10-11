from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set

import pandas as pd
import streamlit as st

from amazon_product_search import query_builder, source
from amazon_product_search.es_client import EsClient
from amazon_product_search.models.search import RequestParams, Response, Result

es_client = EsClient(
    es_host="http://localhost:9200",
)


@dataclass
class Variant:
    name: str
    use_description: bool
    top_k: int


@st.cache
def load_labels(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def compute_recall(retrieved_ids: Iterable[str], relevant_ids: Set[str]) -> int:
    for retrieved_id in retrieved_ids:
        if retrieved_id in relevant_ids:
            return 1
    return 0


def compute_map(retrieved_ids: Iterable[str], relevant_ids: Set[str]) -> float:
    for i, retrieved_id in enumerate(retrieved_ids):
        if retrieved_id in relevant_ids:
            return 1 / (i + 1)
    return 0


def compute_metrics(variant: Variant, query: str, labels_df: pd.DataFrame) -> Dict[str, Any]:
    params = RequestParams(
        query=query,
        use_description=variant.use_description,
    )
    es_query = query_builder.build(params)

    es_response = es_client.search(index_name="products_jp", es_query=es_query, size=variant.top_k)

    response = Response(
        results=[Result(product=hit["_source"], score=hit["_score"]) for hit in es_response["hits"]["hits"]],
        total_hits=es_response["hits"]["total"]["value"],
    )
    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = labels_df[labels_df["esci_label"] == "exact"]["product_id"].tolist()
    return {
        "variant": variant.name,
        "total_hits": response.total_hits,
        "recall": compute_recall(retrieved_ids, relevant_ids),
        "map": compute_map(retrieved_ids, relevant_ids),
    }


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")

    st.write("## Offline Experiment")

    locale = "jp"
    nrows = 1000

    variants = [
        Variant(name="title", use_description=False, top_k=100),
        Variant(name="title_description", use_description=True, top_k=100),
    ]
    st.write("#### Variants")
    st.write(variants)

    clicked = st.button("Run")

    if not clicked:
        return

    labels_df = load_labels(locale, nrows)
    query_dict: Dict[str, pd.DataFrame] = {}
    for query, query_labels_df in labels_df.groupby("query"):
        query_dict[query] = query_labels_df

    total_examples = len(variants) * len(query_dict)
    n = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    metrics = []
    for variant in variants:
        for query, query_labels_df in query_dict.items():
            progress_text.text(f"Query ({n} / {total_examples}): {query}")
            n += 1
            progress_bar.progress(n / total_examples)
            metrics.append(compute_metrics(variant, query, query_labels_df))
    progress_text.text(f"Done ({n} / {total_examples})")

    st.write("#### Result")
    metrics_df = (
        pd.DataFrame(metrics)
        .groupby("variant")
        .agg(
            {
                "total_hits": "mean",
                "recall": "mean",
                "map": "mean",
            }
        )
        .reset_index()
    )
    st.write(metrics_df)


if __name__ == "__main__":
    main()
