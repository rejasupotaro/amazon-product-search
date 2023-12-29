from copy import deepcopy

import streamlit as st

from amazon_product_search.constants import HF
from amazon_product_search.core.source import Locale
from amazon_product_search.core.vespa.vespa_client import VespaClient
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder
from demo.apps.search.search_ui import (
    draw_products,
)
from demo.page_config import set_page_config

client = VespaClient()


@st.cache_resource
def get_encoder(locale: Locale) -> SBERTEncoder:
    hf_model_name = {
        "us": HF.EN_MULTIQA,
        "jp": HF.JP_SLUKE_MEAN,
    }[locale]
    return SBERTEncoder(hf_model_name)


def main() -> None:
    set_page_config()
    st.write("## Search")

    with st.sidebar:
        locale = st.selectbox("Locale", ["us", "jp"])
    encoder = get_encoder(locale)

    st.write("#### Input")
    with st.form("input"):
        query_str = st.text_input("Query:")
        rank_profile = st.selectbox("Rank profile", ["ranking_base", "lexical", "semantic", "hybrid"])
        size = st.number_input("size", value=100)
        is_semantic_search_enabled = st.checkbox("Semantic Search Enabled:", value=True)

        if not st.form_submit_button("Search"):
            return

    query = {
        "query": query_str,
        "input.query(titleWeight)": 1.0,
        "input.query(descriptionWeight)": 1.0,
        "ranking.profile": rank_profile,
        "hits": size,
    }
    if is_semantic_search_enabled:
        query[
            "yql"
        ] = """
        select
            *
        from
            product
        where
            userQuery()
            or ({targetHits:100, approximate:true}nearestNeighbor(product_vector, query_vector))
        """
        query_vector = encoder.encode(query_str)
        query_vector = [float(v) for v in query_vector]
        query["input.query(query_vector)"] = query_vector
    else:
        query[
            "yql"
        ] = """
        select
            *
        from
            product
        where
            userQuery()
        """

    _query = deepcopy(query)
    if is_semantic_search_enabled:
        del _query["input.query(query_vector)"]
        _query["input.query(query_vector)"] = "..."
    st.write(_query)

    st.write("----")

    st.write("#### Output")
    response = client.search(query)
    header = f"{response.total_hits} products found"
    st.write(header)
    draw_products(response.results, {})


if __name__ == "__main__":
    main()
