import re

import pandas as pd
import streamlit as st
from more_itertools import chunked
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.nlp.extractor import KeywordExtractor
from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer
from demo.page_config import set_page_config
from demo.utils import load_products

tokenizer = Tokenizer()
extractor = KeywordExtractor()


def draw_results(results: dict[str, list[tuple[str, float]]]):
    rows = []
    for result in list(zip(*results.values())):
        row = {}
        for method, (keyword, score) in zip(results.keys(), result):
            row[method] = (keyword, round(score, 4))
        rows.append(row)
    st.write(pd.DataFrame(rows))


def main():
    set_page_config()
    st.write("## Keyword Extraction")

    st.write("### Product Catalogue")
    df = load_products(locale="jp", nrows=1000)
    df = df[~df["product_description"].isna() & ~df["product_bullet_point"].isna()]
    df = df.fillna("")

    product = None

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df, gridOptions=grid_options).selected_rows

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
        "yake": extractor.apply_yake(text),
        "position_rank": extractor.apply_position_rank(text),
        "multipartite_rank": extractor.apply_multipartite_rank(text),
        "keybert": extractor.apply_keybert(text),
    }
    draw_results(results)

    st.write("### Highlight")
    for methods in chunked(list(results.keys()), 2):
        columns = st.columns(2)
        for i, method in enumerate(methods):
            with columns[i]:
                st.write(f"#### {method}")
                for keyword, score in results[method]:
                    text = re.sub(keyword, f"<mark style='background-color:#FF9900'>{keyword}</mark>", text)
                st.markdown(text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
