from typing import Any

import streamlit as st

from amazon_product_search.dense_retrieval.retriever import Retriever


def draw_products(products: list[Any]):
    for product in products:
        st.write(product)
        st.write("----")


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")
    retriever = Retriever()

    query = st.text_input("Query:")
    if not query:
        return

    products = retriever.search(query)
    draw_products(products)


if __name__ == "__main__":
    main()
