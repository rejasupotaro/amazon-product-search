from typing import Any, Optional

import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.metrics import compute_ndcg, compute_precision, compute_recall
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking.reranker import from_string
from demo.apps.search.search_ui import draw_input_form, draw_products, draw_response_stats
from demo.page_config import set_page_config
from demo.utils import load_merged, split_fields

es_client = EsClient()
query_builder = QueryBuilder()


@st.cache_data
def load_dataset() -> dict[str, dict[str, tuple[str, str]]]:
    df = load_merged(locale="jp").to_pandas()
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (row["esci_label"], row["product_title"])
    return query_to_label


def draw_es_query(query: Optional[dict[str, Any]], knn_query: Optional[dict[str, Any]], size: int):
    es_query: dict[str, Any] = {
        "size": size,
    }

    if query:
        es_query["query"] = query

    if knn_query:
        es_query["knn"] = knn_query

    st.write("Elasticsearch Query:")
    st.write(es_query)


def main():
    set_page_config()
    st.write("## Search")

    queries, query_to_label = None, {}
    use_dataset = st.checkbox("Use Dataset:", value=True)
    if use_dataset:
        query_to_label = load_dataset()
        queries = query_to_label.keys()

    st.write("#### Input")
    with st.form("input"):
        form_input = draw_input_form(es_client.list_indices(), queries)
        if not st.form_submit_button("Search"):
            return

    size = 20
    normalized_query = normalize_query(form_input.query)
    sparse_fields, dense_fields = split_fields(form_input.fields)
    sparse_query = None
    if sparse_fields:
        sparse_query = query_builder.build_sparse_search_query(
            query=normalized_query,
            fields=sparse_fields,
            query_type=form_input.query_type,
            boost=form_input.sparse_boost,
            is_synonym_expansion_enabled=form_input.is_synonym_expansion_enabled,
        )
    dense_query = None
    if normalized_query and dense_fields:
        dense_query = query_builder.build_dense_search_query(
            normalized_query, field=dense_fields[0], top_k=size, boost=form_input.dense_boost,
        )
    reranker = from_string(form_input.reranker_str)

    st.write("----")

    with st.expander("Query Details", expanded=False):
        st.write("Normalized Query:")
        st.write(normalized_query)

        st.write("Analyzed Query")
        analyzed_query = es_client.analyze(normalized_query)
        st.write(analyzed_query)

        draw_es_query(sparse_query, dense_query, size)

    label_dict = query_to_label.get(form_input.query, {})
    if label_dict:
        with st.expander("Labels", expanded=False):
            st.write(label_dict)

    st.write("----")

    st.write("#### Output")
    response = es_client.search(
        index_name=form_input.index_name, query=sparse_query, knn_query=dense_query, size=size, explain=True,
    )
    if not response.results:
        return
    response.results = reranker.rerank(normalized_query, response.results)

    query_vector = query_builder.encode(normalized_query)
    draw_response_stats(response, query_vector)

    header = f"{response.total_hits} products found"
    if label_dict:
        retrieved_ids = [result.product["product_id"] for result in response.results]
        judgements = {product_id: label for product_id, (label, product_title) in label_dict.items()}
        relevant_ids = {product_id for product_id, (label, product_title) in label_dict.items() if label == "E"}
        precision = compute_precision(retrieved_ids, relevant_ids)
        recall = compute_recall(retrieved_ids, relevant_ids)
        ndcg = compute_ndcg(retrieved_ids, judgements)
        header = f"{header} (Precision: {precision}, Recall: {recall}, NDCG: {ndcg})"
    st.write(header)
    draw_products(response.results, label_dict)


if __name__ == "__main__":
    main()
