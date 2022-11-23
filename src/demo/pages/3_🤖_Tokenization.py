import streamlit as st

from amazon_product_search.nlp.tokenizer import DicType, OutputFormat, Tokenizer
from demo.page_config import set_page_config
from demo.utils import load_products

unidic_tokenizer = Tokenizer(DicType.UNIDIC, output_format=OutputFormat.DUMP)
ipadic_tokenizer = Tokenizer(DicType.IPADIC, output_format=OutputFormat.DUMP)


def main():
    set_page_config()
    st.write("## Tokenization")
    df = load_products(locale="jp", nrows=1)

    s = df.iloc[0]["bullet_point"]
    st.write(unidic_tokenizer.tokenize(s))
    st.write(ipadic_tokenizer.tokenize(s))


if __name__ == "__main__":
    main()
