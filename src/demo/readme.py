import streamlit as st

from demo.page_config import set_page_config


def main():
    set_page_config()

    content = """
    ## README

    - README
    - Dataset
    - Experiment
    - Search
    """
    st.write(content)


if __name__ == "__main__":
    main()
