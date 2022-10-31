import pandas as pd
import streamlit as st

from amazon_product_search import source


@st.cache
def load_products(locale: str, nrows: int = 1000) -> pd.DataFrame:
    return source.load_products(locale, nrows)


@st.cache
def load_labels(locale: str, nrows: int = 1000) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def draw():
    locale = st.selectbox("Locale:", ["jp", "us", "es"])

    labels_df = load_labels(locale)
    products_df = load_products(locale)

    df = labels_df.merge(products_df, on="product_id", how="left")

    st.text(f"{len(df)} labels found")
    st.dataframe(df)
