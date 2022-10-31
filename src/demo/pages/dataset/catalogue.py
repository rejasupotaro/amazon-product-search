import pandas as pd
import streamlit as st

from amazon_product_search import source
from demo.pages.dataset.utils import analyze_dataframe


@st.cache
def load_products(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_products(locale, nrows)


def draw():
    nrows = 1000

    st.write("## Catalogue")
    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    products_df = load_products(locale, nrows=nrows)

    st.write("### Columns")
    analyzed_df = analyze_dataframe(products_df)
    st.write(analyzed_df)

    st.write("### Examples")
    st.write(products_df)
