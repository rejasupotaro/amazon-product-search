import pandas as pd
import streamlit as st

from amazon_product_search import source


@st.cache
def load_products(locale: str, nrows: int = 1000) -> pd.DataFrame:
    return source.load_products(locale, nrows)


def draw():
    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    products_df = load_products(locale)
    st.text(f"{len(products_df)} products found")
    st.dataframe(products_df)
