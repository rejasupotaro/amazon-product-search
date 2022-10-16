from typing import Any

import streamlit as st

from amazon_product_search.dense_retrieval.retriever import Retriever

retriever = Retriever()


def draw_products(products: list[Any]):
    for product in products:
        st.write(product)
        st.write("----")


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")

    query = st.text_input("Query:")
    if not query:
        return

    response = retriever.search(query, top_k=5)
    draw_products([result.product for result in response.results])


if __name__ == "__main__":
    main()
