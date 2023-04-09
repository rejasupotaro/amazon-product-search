import plotly.express as px
import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from demo.page_config import set_page_config
from demo.utils import analyze_dataframe, load_products


def draw_column_info(products_df: pl.DataFrame):
    st.write("### Columns")
    analyzed_df = analyze_dataframe(products_df)
    st.write(analyzed_df.to_pandas())


def draw_brand_info(products_df: pl.DataFrame):
    st.write("### Brand")
    count_df = products_df.groupby("product_brand").count().sort("count", reverse=True).head(100)

    fig = px.bar(count_df.to_pandas(), x="product_brand", y="count")
    fig.update_layout(title="Top 50 brands")
    st.plotly_chart(fig, use_container_width=True)


def draw_color_info(products_df: pl.DataFrame):
    st.write("### Color")
    count_df = products_df.groupby("product_color").count().sort("count", reverse=True).head(100)

    fig = px.bar(count_df.to_pandas(), x="product_color", y="count")
    fig.update_layout(title="Top 50 color names")
    st.plotly_chart(fig, use_container_width=True)


def draw_examples(products_df: pl.DataFrame):
    st.write("### Examples")

    gb = GridOptionsBuilder.from_dataframe(products_df.to_pandas())
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(products_df.to_pandas(), gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    judgement = selected_rows[0]
    del judgement["_selectedRowNodeInfo"]
    st.write(judgement)


def main():
    set_page_config()
    st.write("## Product Catalogue")

    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    products_df = load_products(locale)
    draw_column_info(products_df)
    draw_brand_info(products_df)
    draw_color_info(products_df)
    draw_examples(products_df.head(1000).to_pandas())


if __name__ == "__main__":
    main()
