import pandas as pd
import streamlit as st

from amazon_product_search import source
from demo.pages.dataset.utils import analyze_dataframe


@st.cache
def load_products(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_products(locale, nrows)


@st.cache
def load_labels(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def draw():
    nrows = 1000

    st.write("## Judgements")
    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    labels_df = load_labels(locale, nrows)
    products_df = load_products(locale, nrows)
    df = labels_df.merge(products_df, on="product_id", how="left")

    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df)

    st.write("### Examples")
    selected_query = st.selectbox("query:", df["query"].unique())
    columns = ["esci_label"] + products_df.columns.to_list()
    selected_df = df[df["query"] == selected_query][columns]
    st.write(selected_df)
    st.write(selected_df.to_dict("records"))
