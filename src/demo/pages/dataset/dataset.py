import pandas as pd
import streamlit as st

from amazon_product_search import source
from demo.page_config import set_page_config
from demo.pages.dataset import catalogue, judgements


@st.cache
def load_products(locale: str, nrows: int = -1) -> pd.DataFrame:
    return source.load_products(locale, nrows)


def main():
    set_page_config()

    with st.sidebar:
        datasets_to_funcs = {
            "Catalogue": catalogue.draw,
            "Judgements": judgements.draw,
        }
        selected_dataset = st.selectbox("Dataset:", datasets_to_funcs.keys())
        locale = st.selectbox("Locale:", ["jp", "us", "es"])

    products_df = load_products(locale)

    datasets_to_funcs[selected_dataset](locale, products_df)


if __name__ == "__main__":
    main()
