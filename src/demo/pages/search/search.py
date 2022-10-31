from typing import Any, Optional

import streamlit as st

from amazon_product_search.es import query_builder
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.response import Result
from amazon_product_search.nlp.encoder import Encoder
from demo.page_config import set_page_config

es_client = EsClient()
encoder = Encoder()


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


def main():
    set_page_config()

    size = 20

    st.write("## Search")

    st.write("#### Input")
    indices = es_client.list_indices()
    index_name = st.selectbox("Index:", indices)

    query = st.text_input("Query:")

    cols = st.columns(2)
    is_sparse_enabled = cols[0].checkbox("Sparse:", value=True)
    is_dense_enabled = cols[1].checkbox("Dense:", value=False)

    use_description = False
    use_bullet_point = False
    use_brand = False
    use_color_name = False
    if is_sparse_enabled:
        use_description = st.checkbox("Use description")
        use_bullet_point = st.checkbox("Use bullet point")
        use_brand = st.checkbox("Use brand")
        use_color_name = st.checkbox("Use color name")

    es_query = None
    if is_sparse_enabled:
        es_query = query_builder.build_multimatch_search_query(
            query=query,
            use_description=use_description,
            use_bullet_point=use_bullet_point,
            use_brand=use_brand,
            use_color_name=use_color_name,
        )
    es_knn_query = None
    if query and is_dense_enabled:
        query_vector = encoder.encode(query, show_progress_bar=False)
        es_knn_query = query_builder.build_knn_search_query(query_vector, top_k=size)

    draw_es_query(es_query, es_knn_query, size)

    st.write("#### Output")
    response = es_client.search(index_name=index_name, query=es_query, knn_query=es_knn_query, size=size)
    st.write(f"{response.total_hits} products found")
    draw_products(response.results)


if __name__ == "__main__":
    main()
