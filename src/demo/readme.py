import streamlit as st

from demo.page_config import set_page_config


def main():
    set_page_config()

    content = """
    ## README

    - README
    - Dataset
    - Sparse Search
    - Offline Experiment
    """
    st.write(content)


if __name__ == "__main__":
    main()
