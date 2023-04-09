import plotly.express as px
import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from demo.page_config import set_page_config
from demo.utils import analyze_dataframe, load_merged


def draw_column_info(df: pl.DataFrame):
    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df.to_pandas())


def draw_label_distribution(df: pl.DataFrame):
    count_df = df.groupby("esci_label").count().sort("count", reverse=True)
    fig = px.bar(count_df.to_pandas(), x="esci_label", y="count")
    fig.update_layout(title="The number of labels")
    st.plotly_chart(fig)


def draw_examples(df: pl.DataFrame):
    st.write("### Examples")

    gb = GridOptionsBuilder.from_dataframe(df.to_pandas())
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df.to_pandas(), gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    judgement = selected_rows[0]
    del judgement["_selectedRowNodeInfo"]
    st.write(judgement)


def main():
    set_page_config()
    st.write("## Relevance Judgements")

    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    df = load_merged(locale)
    draw_column_info(df)
    draw_label_distribution(df)

    queries = df.get_column("query").unique().to_list()
    query = st.selectbox("Query:", queries)
    filtered_df = df.filter(pl.col("query") == query)
    draw_examples(filtered_df)


if __name__ == "__main__":
    main()
