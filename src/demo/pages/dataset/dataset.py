import streamlit as st

from demo.page_config import set_page_config
from demo.pages.dataset import catalogue, judgements


def main():
    set_page_config()

    with st.sidebar:
        datasets_to_funcs = {
            "Catalogue": catalogue.draw,
            "Judgements": judgements.draw,
        }
        selected_dataset = st.selectbox("Dataset:", datasets_to_funcs.keys())

    datasets_to_funcs[selected_dataset]()


if __name__ == "__main__":
    main()
