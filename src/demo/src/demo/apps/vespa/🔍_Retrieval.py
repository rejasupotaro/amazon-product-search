from copy import deepcopy

import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from data_source import Locale, loader

from amazon_product_search.constants import HF
from amazon_product_search.retrieval.response import Response
from amazon_product_search.vespa.query_builder import QueryBuilder
from amazon_product_search.vespa.vespa_client import VespaClient
from demo.apps.es.search_ui import (
    draw_products,
)
from demo.page_config import set_page_config

client = VespaClient()


@st.cache_resource
def get_query_builder(locale: Locale) -> QueryBuilder:
    hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]
    return QueryBuilder(locale=locale, hf_model_name=hf_model_name)


@st.cache_data
def load_dataset(locale: Locale) -> dict[str, dict[str, tuple[str, str]]]:
    df = loader.load_merged(locale=locale)
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (
                row["esci_label"],
                row["product_title"],
            )
    return query_to_label


def draw_response_stats(response: Response, label_dict: dict[str, tuple[str, str]]) -> None:
    rows = []
    for rank, result in enumerate(response.results):
        row = {
            "rank": rank,
            "score": result.score,
            "lexical_score": result.lexical_score,
            "semantic_score": result.semantic_score,
            "label": label_dict.get(result.product["product_id"], ("-", ""))[0],
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    with st.expander("Response Stats"):
        st.write(df)
        cols = st.columns(2)
        with cols[0]:
            plot = sns.jointplot(data=df, x="lexical_score", y="semantic_score", hue="label", height=5)
            st.pyplot(plot.figure)
        with cols[1]:
            fig = px.line(df, x="rank", y="score")
            fig.update_layout(title="Scores by Rank")
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    set_page_config()
    st.write("## Search")

    with st.sidebar:
        locale = st.selectbox("Locale", ["us", "jp"])
        queries, query_to_label = None, {}
        query_to_label = load_dataset(locale)
        queries = query_to_label.keys()

    st.write("#### Input")
    with st.form("input"):
        query_str = st.selectbox("Query:", queries) if queries else st.text_input("Query:")
        rank_profile = st.selectbox("Rank profile", ["hybrid", "lexical", "semantic"])
        size = st.number_input("size", value=100)

        if not st.form_submit_button("Search"):
            return

    is_semantic_search_enabled = rank_profile != "lexical"
    query = get_query_builder(locale).build_query(query_str, rank_profile, size, is_semantic_search_enabled)

    _query = deepcopy(query)
    if "input.query(query_vector)" in _query:
        _query["input.query(query_vector)"] = "..."
    st.write(_query)

    st.write("----")

    st.write("#### Output")
    response = client.search(query)

    label_dict = query_to_label.get(query_str, {})
    draw_response_stats(response, label_dict)

    header = f"{response.total_hits} products found"
    st.write(header)
    draw_products(response.results, label_dict)


if __name__ == "__main__":
    main()
