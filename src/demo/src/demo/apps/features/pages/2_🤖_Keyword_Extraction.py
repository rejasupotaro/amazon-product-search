import re

import polars as pl
import streamlit as st
from more_itertools import chunked
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizers.tokenizer import Tokenizer
from amazon_product_search.retrieval.keyword_extractor import KeywordExtractor
from demo.page_config import set_page_config
from demo.utils import load_products

tokenizer = Tokenizer()
extractor = KeywordExtractor()


def draw_results(results: dict[str, list[tuple[str, float]]]) -> None:
    rows = []
    for result in list(zip(*results.values(), strict=True)):
        row = {}
        for method, (keyword, score) in zip(results.keys(), result, strict=True):
            row[method] = f"{keyword} ({round(score, 4)})"
        rows.append(row)
    st.write(pl.from_dicts(rows).to_pandas())


def main() -> None:
    set_page_config()
    st.write("## Keyword Extraction")

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
    st.write(product["product_title"])

    st.write("#### Input Text")
    text = product["product_description"] + " " + product["product_bullet_point"]
    st.markdown(text, unsafe_allow_html=True)

    st.write("#### Normalized Text")
    text = normalize_doc(text)
    text = " ".join(tokenizer.tokenize(text))
    st.write(text)

    st.write("----")

    st.write("## Results")
    results = {
        "keybert": extractor.apply_keybert(text),
    }
    draw_results(results)

    st.write("### Highlight")
    for methods in chunked(list(results.keys()), 2):
        columns = st.columns(2)
        for i, method in enumerate(methods):
            with columns[i]:
                st.write(f"#### {method}")
                for keyword, _score in results[method]:
                    text = re.sub(
                        keyword,
                        f"<mark style='background-color:#FF9900'>{keyword}</mark>",
                        text,
                    )
                st.markdown(text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
