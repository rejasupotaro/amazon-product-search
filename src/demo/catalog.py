import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

DATA_DIR = "./data"


@st.cache
def load_products(nrows=100) -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/product_catalogue-v0.3.csv.zip", nrows=nrows)


def main():
    st.set_page_config(layout="wide")
    products_df = load_products()
    st.text(f"{len(products_df)} products found")
    AgGrid(products_df)


if __name__ == "__main__":
    main()
