import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

DATA_DIR = "./data"


@st.cache
def load_products(locale: str, nrows: int=100) -> pd.DataFrame:
    filename = f"{DATA_DIR}/product_catalogue-v0.3_{locale}.csv.zip"
    return pd.read_csv(filename, nrows=nrows)


def main():
    st.set_page_config(layout="wide")
    locale = st.selectbox("Locale", ["us", "jp", "es"])
    products_df = load_products(locale)
    st.text(f"{len(products_df)} products found")
    AgGrid(products_df)


if __name__ == "__main__":
    main()
