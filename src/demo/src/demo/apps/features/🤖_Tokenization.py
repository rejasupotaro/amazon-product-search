import streamlit as st
from pandas import DataFrame
from st_aggrid import AgGrid, GridOptionsBuilder

from amazon_product_search.nlp.tokenizers.japanese_tokenizer import DicType, JapaneseTokenizer, OutputFormat
from demo.page_config import set_page_config
from demo.utils import load_products

unidic_tokenizer = JapaneseTokenizer(DicType.UNIDIC, output_format=OutputFormat.DUMP)
ipadic_tokenizer = JapaneseTokenizer(DicType.IPADIC, output_format=OutputFormat.DUMP)


def main() -> None:
    set_page_config()
    st.write("## Tokenization")

    st.write("### Product Catalogue")
    df = load_products(locale="jp", nrows=1000)
    df = df[(~df["product_description"].isnull()) & (~df["product_bullet_point"].isnull())]
    df = df.fillna("")

    product = None

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df, gridOptions=grid_options).selected_rows

    if selected_rows is None or len(selected_rows) <= 0:
        return

    product = selected_rows.to_dict("records")[0]

    st.write("### Input Text")
    s = product["product_title"]
    st.write(s)

    columns = st.columns(2)
    with columns[0]:
        st.write("### Fugashi (UniDic)")
        st.write(
            DataFrame([{"token": token, "pos": pos} for token, pos in unidic_tokenizer.tokenize(s)])
        )
    with columns[1]:
        st.write("### Fugashi (IPADIC)")
        st.write(
            DataFrame([{"token": token, "pos": list(pos)} for token, pos in ipadic_tokenizer.tokenize(s)])
        )


if __name__ == "__main__":
    main()
