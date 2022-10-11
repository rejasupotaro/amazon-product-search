import streamlit as st

from demo import dataset, experiment, readme, sparse_search


def main():
    st.set_page_config(layout="wide")

    pages_to_funcs = {
        "README": readme.main,
        "Dataset": dataset.main,
        "Sparse Search": sparse_search.main,
        "Offline Experiment": experiment.main,
    }

    selected_page = st.sidebar.selectbox("Select Page", pages_to_funcs.keys())
    pages_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
