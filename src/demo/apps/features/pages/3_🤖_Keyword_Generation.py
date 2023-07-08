import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.core.nlp.normalizer import normalize_doc
from amazon_product_search.core.nlp.tokenizer import Tokenizer
from amazon_product_search.core.retrieval.keyword_generator import KeywordGenerator
from demo.page_config import set_page_config
from demo.utils import load_products

tokenizer = Tokenizer()
generator = KeywordGenerator()


def draw_results(results: dict[str, list[tuple[str, float]]]) -> None:
    rows = []
    for result in list(zip(*results.values(), strict=True)):
        row = {}
        for method, (keyword, score) in zip(results.keys(), result, strict=True):
            row[method] = (keyword, round(score, 4))
        rows.append(row)
    st.write(pl.from_dicts(rows).to_pandas())


def main() -> None:
    set_page_config()
    st.write("## Keyword Generation")

    st.write("### Product Catalogue")
    df = load_products(locale="jp", nrows=1000)
    df = df.filter((pl.col("product_description").is_not_null() & pl.col("product_bullet_point").is_not_null()))
    df = df.fill_null("")

    product = None

    gb = GridOptionsBuilder.from_dataframe(df.to_pandas())
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df.to_pandas(), gridOptions=grid_options).selected_rows

    if not selected_rows:
        return

    product = selected_rows[0]
    del product["_selectedRowNodeInfo"]

    st.write("----")

    st.write("### Selected Product")

    st.write("#### Title")
    title = product["product_title"]
    st.write(title)
    normalized_text = normalize_doc(title)
    st.write(generator.generate(normalized_text))

    st.write("#### Description")
    description = product["product_description"]
    st.markdown(description, unsafe_allow_html=True)
    normalized_text = normalize_doc(description)
    st.write(generator.generate(normalized_text))

    st.write("#### Bullet Point")
    bullet_point = product["product_bullet_point"]
    st.write(bullet_point)
    normalized_text = normalize_doc(bullet_point)
    st.write(generator.generate(normalized_text))


if __name__ == "__main__":
    main()
