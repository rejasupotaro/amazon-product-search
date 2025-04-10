import plotly.express as px
import streamlit as st
from pandas import DataFrame
from st_aggrid import AgGrid, GridOptionsBuilder

from demo.page_config import set_page_config
from demo.utils import analyze_dataframe, load_products


def draw_column_info(products_df: DataFrame) -> None:
    st.write("### Columns")
    analyzed_df = analyze_dataframe(products_df)
    st.write(analyzed_df)


def draw_brand_info(products_df: DataFrame) -> None:
    st.write("### Brand")
    count_df = products_df.groupby("product_brand").size().reset_index(name="count").sort_values("count", ascending=False).head(100)

    fig = px.bar(count_df, x="product_brand", y="count")
    fig.update_layout(title="Top 50 brands")
    st.plotly_chart(fig, use_container_width=True)


def draw_color_info(products_df: DataFrame) -> None:
    st.write("### Color")
    count_df = products_df.groupby("product_color").size().reset_index(name="count").sort_values("count", ascending=False).head(100)

    fig = px.bar(count_df, x="product_color", y="count")
    fig.update_layout(title="Top 50 color names")
    st.plotly_chart(fig, use_container_width=True)


def draw_examples(products_df: DataFrame) -> None:
    st.write("### Examples")

    gb = GridOptionsBuilder.from_dataframe(products_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(products_df, gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    product = selected_rows[0]
    del product["_selectedRowNodeInfo"]
    for key, value in product.items():
        st.write(f"#### {key}")
        st.text(value)


def main() -> None:
    set_page_config()
    st.write("## Product Catalogue")

    with st.sidebar:
        locale = st.selectbox("Locale:", ["us", "jp", "es"])

    products_df = load_products(locale)
    draw_column_info(products_df)
    draw_brand_info(products_df)
    draw_color_info(products_df)
    draw_examples(products_df.head(1000))


if __name__ == "__main__":
    main()
