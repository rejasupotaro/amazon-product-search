import pandas as pd
import plotly.express as px
import streamlit as st

from demo.utils import analyze_dataframe


def draw_column_info(products_df: pd.DataFrame):
    st.write("### Columns")
    analyzed_df = analyze_dataframe(products_df)
    st.write(analyzed_df)


def draw_brand_info(products_df: pd.DataFrame):
    st.write("### Brand")
    count_df = products_df.groupby("product_brand").size().reset_index(name="count")
    count_df = count_df.sort_values(by="count", ascending=False).head(100)

    fig = px.bar(count_df, x="product_brand", y="count")
    fig.update_layout(title="Top 50 brands")
    st.plotly_chart(fig, use_container_width=True)


def draw_color_info(products_df: pd.DataFrame):
    st.write("### Color")
    count_df = products_df.groupby("product_color_name").size().reset_index(name="count")
    count_df = count_df.sort_values(by="count", ascending=False).head(100)

    fig = px.bar(count_df, x="product_color_name", y="count")
    fig.update_layout(title="Top 50 color names")
    st.plotly_chart(fig, use_container_width=True)


def draw_examples(products_df: pd.DataFrame):
    st.write("### Examples")
    st.write(products_df.head(100))


def draw(locale: str, products_df: pd.DataFrame):
    st.write("## Catalogue")
    draw_column_info(products_df)
    draw_brand_info(products_df)
    draw_color_info(products_df)
    draw_examples(products_df)
