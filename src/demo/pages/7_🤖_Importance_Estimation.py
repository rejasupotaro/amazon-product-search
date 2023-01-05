from typing import Optional

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.retrieval.importance_estimator import ColBERTTermImportanceEstimator
from demo.page_config import set_page_config
from demo.utils import load_products

estimator = ColBERTTermImportanceEstimator()


def estimate_importance(text: Optional[str]):
    if not text:
        return
    results = estimator.estimate(text)
    results = sorted(results, key=lambda result: result[1], reverse=True)
    st.write([f"{result[0]} ({result[1]})" for result in results])


def main():
    set_page_config()
    st.write("## Importance Estimation")

    products_df = load_products("jp").head(20)

    st.write("### Products")

    gb = GridOptionsBuilder.from_dataframe(products_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(products_df, gridOptions=grid_options).selected_rows

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
