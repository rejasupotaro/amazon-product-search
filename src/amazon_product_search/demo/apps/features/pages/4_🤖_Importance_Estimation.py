from typing import Optional

import plotly.express as px
import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.core.retrieval.importance_estimator import (
    ColBERTTermImportanceEstimator,
)
from amazon_product_search.demo.page_config import set_page_config
from amazon_product_search.demo.utils import load_products

estimator = ColBERTTermImportanceEstimator()


def estimate_importance(text: Optional[str]) -> None:
    if not text:
        return
    results = estimator.estimate(text)

    weights_df = pl.from_dict(
        {
            "token": [r[0] for r in results],
            "weight": [r[1] for r in results],
        }
    )
    fig = px.bar(weights_df.to_pandas(), x="token", y="weight")
    st.plotly_chart(fig)

    results = sorted(results, key=lambda result: result[1], reverse=True)
    st.write([f"{result[0]} ({result[1]})" for result in results])


def main() -> None:
    set_page_config()
    st.write("## Importance Estimation")

    products_df = load_products("jp").head(20)

    st.write("### Products")

    gb = GridOptionsBuilder.from_dataframe(products_df.to_pandas())
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(products_df.to_pandas(), gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    st.write("### Selected Product")
    row = selected_rows[0]
    del row["_selectedRowNodeInfo"]
    st.write(row)

    st.write("### Estimated Term Importance")
    st.write("#### Title")
    title = row["product_title"]
    estimate_importance(title)
    st.write("#### Description")
    description = row["product_description"]
    estimate_importance(description)
    st.write("#### Bullet Point")
    bullet_point = row["product_bullet_point"]
    estimate_importance(bullet_point)


if __name__ == "__main__":
    main()
