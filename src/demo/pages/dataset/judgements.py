from typing import Any

import pandas as pd
import streamlit as st

from amazon_product_search import source
from demo.pages.dataset.utils import analyze_dataframe


@st.cache
def load_labels(locale: str, nrows: int = -1) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def draw_column_info(df: pd.DataFrame):
    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df)


def draw_examples(df: pd.DataFrame, columns_to_show: list[str]):
    st.write("### Examples")
    selected_query = st.selectbox("query:", df["query"].unique())
    selected_df = df[df["query"] == selected_query][columns_to_show]

    st.write("judgements:")
    st.write(selected_df)
    draw_products(selected_df.to_dict("records"))


def draw_products(products: list[dict[str, Any]]):
    for product in products:
        st.write("----")
        st.write(f'[{product["esci_label"]}]')
        for column in [
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color_name",
        ]:
            st.write(f"##### {column}")
            st.markdown(product[column], unsafe_allow_html=True)


def draw(locale: str, products_df: pd.DataFrame):
    st.write("## Judgements")
    labels_df = load_labels(locale, nrows=100)
    df = labels_df.merge(products_df, on="product_id", how="left")

    draw_column_info(df)
    draw_examples(df, ["esci_label"] + products_df.columns.to_list())
