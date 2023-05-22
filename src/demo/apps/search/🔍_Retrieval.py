from typing import Any, Optional

import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.es.response import Result
from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.reranking.reranker import from_string
from demo.apps.search.search_ui import draw_input_form, draw_response_stats
from demo.page_config import set_page_config
from demo.utils import split_fields

es_client = EsClient()
query_builder = QueryBuilder()


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



def draw_products(results: list[Result]):
    for result in results:
        with st.expander(f"{result.product['product_title']} ({result.score})"):
            st.write(result.product)
            st.write(result.explanation)


def main():
    set_page_config()
    st.write("## Search")

    st.write("#### Input")
    with st.form("input"):
        form_input = draw_input_form(es_client.list_indices())
        if not st.form_submit_button("Search"):
            return

    size = 20
    normalized_query = normalize_query(form_input.query)
    sparse_fields, dense_fields = split_fields(form_input.fields)
    es_query = None
    if sparse_fields:
        es_query = query_builder.build_sparse_search_query(
            query=normalized_query,
            fields=sparse_fields,
            query_type=form_input.query_type,
            boost=form_input.sparse_boost,
            is_synonym_expansion_enabled=form_input.is_synonym_expansion_enabled,
        )
    es_knn_query = None
    if normalized_query and dense_fields:
        es_knn_query = query_builder.build_dense_search_query(
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

        draw_es_query(es_query, es_knn_query, size)

    st.write("----")

    st.write("#### Output")
    response = es_client.search(
        index_name=form_input.index_name, query=es_query, knn_query=es_knn_query, size=size, explain=True,
    )
    if not response.results:
        return
    response.results = reranker.rerank(normalized_query, response.results)

    query_vector = query_builder.encode(normalized_query)
    draw_response_stats(response, query_vector)

    st.write(f"{response.total_hits} products found")
    draw_products(response.results)


if __name__ == "__main__":
    main()
