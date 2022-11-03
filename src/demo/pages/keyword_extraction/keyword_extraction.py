import re

import pandas as pd
import streamlit as st

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
    set_page_config(layout="wide")

    st.write("## Keyword Extraction")

    st.write("### Catalogue")
    df = load_products(locale="jp", nrows=100)
    df = df[~df["product_description"].isna() & ~df["product_bullet_point"].isna()]
    df = df.fillna("")
    st.write(df)
    i = st.slider("i:", min_value=0, max_value=len(df))

    st.write("----")

    st.write("### Selected Product")
    product = df.iloc[i]

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
    }
    draw_results(results)

    st.write("### Highlight")
    method = st.selectbox("Method:", results.keys())
    for keyword, score in results[method]:
        text = re.sub(keyword, f"<mark style='background-color:#FF9900'>{keyword}</mark>", text)
    st.markdown(text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
