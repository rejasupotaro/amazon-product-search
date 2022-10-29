from typing import Any

import streamlit as st

from amazon_product_search.es import query_builder
from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.response import Response, Result

es_client = EsClient(
    es_host="http://localhost:9200",
)


def search(es_query: dict[str, Any], index_name: str) -> Response:
    return es_client.search(index_name=index_name, es_query=es_query)


def draw_products(results: list[Result]):
    for result in results:
        st.write(result.product)
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
    use_description = st.checkbox("Use description")
    use_bullet_point = st.checkbox("Use bullet point")
    use_brand = st.checkbox("Use brand")
    use_color_name = st.checkbox("Use color name")

    st.write("Elasticsearch Query:")
    es_query = query_builder.build_multimatch_search_query(
        query=query,
        use_description=use_description,
        use_bullet_point=use_bullet_point,
        use_brand=use_brand,
        use_color_name=use_color_name,
    )
    st.json(es_query)

    st.write("----")

    st.write("#### Output")
    response = search(es_query, selected_index)
    st.write(f"{response.total_hits} products found")
    draw_products(response.results)


if __name__ == "__main__":
    main()
