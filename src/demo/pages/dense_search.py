from typing import Any

import streamlit as st

from amazon_product_search.dense_retrieval.encoder import Encoder
from amazon_product_search.es import query_builder
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.models.search import Response, Result

encoder = Encoder()
es_client = EsClient()


def draw_products(products: list[Any]):
    for product in products:
        st.write(product)
        st.write("----")


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")

    st.markdown("## Indices")
    indices = es_client.list_indices()
    selected_index = st.selectbox("Index:", indices)

    st.markdown("#### Count")
    count = es_client.count_docs(selected_index)
    st.write(count)

    st.write("## Search")

    st.write("#### Input")
    query = st.text_input("Query:")
    if not query:
        return

    query_vector = encoder.encode(query, show_progress_bar=False)
    es_query = query_builder.build_knn_search_query(query_vector, top_k=20)
    es_response = es_client.knn_search(selected_index, es_query)
    response = Response(
        results=[Result(product=hit["_source"], score=hit["_score"]) for hit in es_response["hits"]["hits"]],
        total_hits=es_response["hits"]["total"]["value"],
    )
    draw_products([result.product for result in response.results])


if __name__ == "__main__":
    main()
