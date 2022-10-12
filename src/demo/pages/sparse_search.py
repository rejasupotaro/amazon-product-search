from typing import Any, Dict, List

import streamlit as st

from amazon_product_search import query_builder
from amazon_product_search.es_client import EsClient
from amazon_product_search.models.search import RequestParams, Response, Result

es_client = EsClient(
    es_host="http://localhost:9200",
)


def search(es_query: Dict[str, Any], index_name: str) -> Response:
    es_response = es_client.search(index_name=index_name, es_query=es_query)
    response = Response(
        results=[Result(product=hit["_source"], score=hit["_score"]) for hit in es_response["hits"]["hits"]],
        total_hits=es_response["hits"]["total"]["value"],
    )
    return response


def draw_products(results: List[Result]):
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
    params = RequestParams(
        query=query,
        use_description=use_description,
        use_bullet_point=use_bullet_point,
        use_brand=use_brand,
        use_color_name=use_color_name,
    )

    st.write("Elasticsearch Query:")
    es_query = query_builder.build(params)
    st.json(es_query)

    st.write("----")

    st.write("#### Output")
    response = search(es_query, selected_index)
    st.write(f"{response.total_hits} products found")
    draw_products(response.results)


if __name__ == "__main__":
    main()
