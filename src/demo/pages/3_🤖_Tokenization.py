import polars as pl
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.nlp.tokenizer import DicType, OutputFormat, Tokenizer
from demo.page_config import set_page_config
from demo.utils import load_products

unidic_tokenizer = Tokenizer(DicType.UNIDIC, output_format=OutputFormat.DUMP)
ipadic_tokenizer = Tokenizer(DicType.IPADIC, output_format=OutputFormat.DUMP)


def main():
    set_page_config()
    st.write("## Tokenization")

    st.write("### Product Catalogue")
    df = load_products(locale="jp", nrows=1000)
    df = df.filter((pl.col("product_description").is_not_null()) & pl.col("product_bullet_point").is_not_null())
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

    st.write("### Input Text")
    s = product["product_title"]
    st.write(s)

    columns = st.columns(2)
    with columns[0]:
        st.write("### Fugashi (UniDic)")
        st.write(
            pl.from_dicts([{"token": token, "pos": pos} for token, pos in unidic_tokenizer.tokenize(s)]).to_pandas()
        )
    with columns[1]:
        st.write("### Fugashi (IPADIC)")
        st.write(
            pl.from_dicts([{"token": token, "pos": pos} for token, pos in ipadic_tokenizer.tokenize(s)]).to_pandas()
        )


if __name__ == "__main__":
    main()
