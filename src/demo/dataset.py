import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

from amazon_product_search import source

DATA_DIR = "./data"


@st.cache
def load_products(locale: str, nrows: int = 100) -> pd.DataFrame:
    return source.load_products(locale, nrows)


@st.cache
def load_labels(locale: str, nrows: int = 100) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def draw_products():
    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    products_df = load_products(locale)
    st.text(f"{len(products_df)} products found")
    AgGrid(products_df)


def draw_labels():
    locale = st.selectbox("Locale:", ["jp", "us", "es"])

    labels_df = load_labels(locale)
    products_df = load_products(locale)

    df = labels_df.merge(products_df, on="product_id", how="left")

    st.text(f"{len(df)} labels found")
    AgGrid(df)


def main():
    datasets_to_funcs = {
        "Products": draw_products,
        "Labels": draw_labels,
    }

    selected_dataset = st.selectbox("Dataset:", datasets_to_funcs.keys())
    datasets_to_funcs[selected_dataset]()


if __name__ == "__main__":
    main()
