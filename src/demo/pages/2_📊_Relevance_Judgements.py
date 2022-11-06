from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from demo.page_config import set_page_config
from demo.utils import analyze_dataframe, load_merged


def draw_column_info(df: pd.DataFrame):
    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df)


def draw_label_distribution(df: pd.DataFrame):
    count_df = df.groupby("esci_label").size().reset_index(name="count")
    count_df = count_df.sort_values("count", ascending=False)
    fig = px.bar(count_df, x="esci_label", y="count")
    fig.update_layout(title="The number of labels")
    st.plotly_chart(fig)


def draw_examples(df: pd.DataFrame):
    st.write("### Examples")

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df, gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    judgement = selected_rows[0]
    del judgement["_selectedRowNodeInfo"]
    st.write(judgement)


def draw_products(products: list[dict[str, Any]]):
    for product in products:
        st.write("----")
        st.write(f'[{product["esci_label"]}]')
        for column in [
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color_name",
        ]:
            st.write(f"##### {column}")
            st.markdown(product[column], unsafe_allow_html=True)


def main():
    set_page_config()

    st.write("## Judgements")
    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    df = load_merged(locale)
    draw_column_info(df)
    draw_label_distribution(df)

    queries = df["query"].unique()
    query = st.selectbox("Query:", queries)
    filtered_df = df[df["query"] == query]
    draw_examples(filtered_df)


if __name__ == "__main__":
    main()
