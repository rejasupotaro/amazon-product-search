import streamlit as st

from demo.page_config import set_page_config
from demo.pages.dataset import catalogue, judgements
from demo.utils import load_products


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
