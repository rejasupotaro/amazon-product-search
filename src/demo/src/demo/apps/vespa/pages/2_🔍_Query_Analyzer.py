import json

import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from data_source import Locale

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


def draw_response_stats(response: Response) -> None:
    rows = []
    for rank, result in enumerate(response.results):
        row = {
            "rank": rank,
            "score": result.score,
            "lexical_score": result.lexical_score,
            "semantic_score": result.semantic_score,
            "label": "-",
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

    st.write("#### Input")
    with st.form("input"):
        query = {
            "ranking.profile": "lexical",
            "yql": "select * from product where default contains 'query'",
            "hits": 100,
        }
        query = st.text_area("Query:", json.dumps(query, indent=4), height=200)

        if not st.form_submit_button("Search"):
            return

    st.write("----")

    st.write("#### Output")
    query = json.loads(query)
    st.write(query)
    response = client.search(query)

    draw_response_stats(response)

    header = f"{response.total_hits} products found"
    st.write(header)
    draw_products(response.results, label_dict={})


if __name__ == "__main__":
    main()
