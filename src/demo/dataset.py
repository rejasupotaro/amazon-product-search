import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

DATA_DIR = "./data"


@st.cache
def load_products(locale: str, nrows: int = 100) -> pd.DataFrame:
    filename = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows)


@st.cache
def load_labels(locale: str, nrows: int = 100) -> pd.DataFrame:
    filename = f"{DATA_DIR}/train-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows)


def draw_products():
    locale = st.selectbox("Locale", ["jp", "us", "es"])
    products_df = load_products(locale)
    st.text(f"{len(products_df)} products found")
    AgGrid(products_df)


def draw_labels():
    locale = st.selectbox("Locale", ["jp", "us", "es"])

    labels_df = load_labels(locale)
    products_df = load_products(locale)

    df = labels_df.merge(products_df, on="product_id", how="left")

    st.text(f"{len(df)} labels found")
    AgGrid(df)


def main():
    st.set_page_config(layout="wide")

    page_names_to_funcs = {
        "Products": draw_products,
        "Labels": draw_labels,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
