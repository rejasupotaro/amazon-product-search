from typing import Any, Optional

import streamlit as st

from amazon_product_search.constants import HF
from amazon_product_search.core import source
from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.metrics import (
    compute_ndcg,
    compute_precision,
    compute_recall,
)
from amazon_product_search.core.reranking.reranker import from_string
from amazon_product_search.core.retrieval.rank_fusion import RankFusion
from amazon_product_search.core.retrieval.retriever import Retriever
from amazon_product_search.core.source import Locale
from demo.apps.search.search_ui import (
    draw_input_form,
    draw_products,
    draw_response_stats,
)
from demo.page_config import set_page_config

es_client = EsClient()


@st.cache_resource
def get_query_builder(locale: Locale) -> QueryBuilder:
    hf_model_name = {
        "us": HF.EN_MULTIQA,
        "jp": HF.JP_SLUKE_MEAN,
    }[locale]
    return QueryBuilder(locale=locale, hf_model_name=hf_model_name)


def get_retriever(locale: Locale, es_client: EsClient, query_builder: QueryBuilder) -> Retriever:
    return Retriever(locale, es_client, query_builder)


@st.cache_data
def load_dataset(locale: Locale) -> dict[str, dict[str, tuple[str, str]]]:
    df = source.load_merged(locale).to_pandas()
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (
                row["esci_label"],
                row["product_title"],
            )
    return query_to_label


def draw_es_query(query: Optional[dict[str, Any]], knn_query: Optional[dict[str, Any]], size: int) -> None:
    es_query: dict[str, Any] = {
        "size": size,
    }

    if query:
        es_query["query"] = query

    if knn_query:
        es_query["knn"] = knn_query

    st.write("Elasticsearch Query:")
    st.write(es_query)


def main() -> None:
    set_page_config()
    st.write("## Search")

    with st.sidebar:
        locale = st.selectbox("Locale", ["us", "jp"])
        index_name = st.selectbox("Index:", es_client.list_indices())

        queries, query_to_label = None, {}
        use_dataset = st.checkbox("Use Dataset:", value=True)
        if use_dataset:
            query_to_label = load_dataset(locale)
            queries = query_to_label.keys()

    st.write("#### Input")
    with st.form("input"):
        form_input = draw_input_form(queries)
        if not st.form_submit_button("Search"):
            return

    response = get_retriever(locale, es_client, get_query_builder(locale)).search(
        index_name=index_name,
        query=form_input.query,
        fields=form_input.fields,
        query_type=form_input.query_type,
        is_synonym_expansion_enabled=form_input.is_synonym_expansion_enabled,
        sparse_boost=form_input.sparse_boost,
        dense_boost=form_input.dense_boost,
        size=form_input.size,
        window_size=form_input.window_size,
        rank_fusion=RankFusion(
            fuser=form_input.fuser,
            combination_method=form_input.combination_method,
            score_transformation_method=form_input.score_transformation_method,
            weighting_strategy=form_input.weighting_strategy,
        ),
    )
    reranker = from_string(form_input.reranker_str)

    st.write("----")

    label_dict: dict[str, str] = query_to_label.get(form_input.query, {})
    if label_dict:
        with st.expander("Labels", expanded=False):
            st.write(label_dict)

    st.write("----")

    st.write("#### Output")
    if not response.results:
        return
    response.results = reranker.rerank(form_input.query, response.results)

    query_vector = get_query_builder(locale).encode(form_input.query)
    draw_response_stats(response, query_vector, label_dict)

    header = f"{response.total_hits} products found"
    if label_dict:
        retrieved_ids = [result.product["product_id"] for result in response.results]
        id_to_label = {product_id: label for product_id, (label, product_title) in label_dict.items()}
        relevant_ids = {product_id for product_id, (label, product_title) in label_dict.items() if label == "E"}
        precision = compute_precision(retrieved_ids, relevant_ids)
        recall = compute_recall(retrieved_ids, relevant_ids)
        ndcg = compute_ndcg(retrieved_ids, id_to_label)
        header = f"{header} (Precision: {precision}, Recall: {recall}, NDCG: {ndcg})"
    st.write(header)
    draw_products(response.results, label_dict)


if __name__ == "__main__":
    main()
