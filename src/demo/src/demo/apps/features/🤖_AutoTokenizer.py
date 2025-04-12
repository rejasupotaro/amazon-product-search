from typing import Any

import streamlit as st
from pandas import DataFrame
from st_aggrid import AgGrid, GridOptionsBuilder
from transformers import AutoTokenizer, PreTrainedTokenizer

from demo.page_config import set_page_config
from demo.utils import load_products


@st.cache_data
def get_tokenizer(name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def draw_tokenizer_info(tokenizer_dict: dict[str, PreTrainedTokenizer]) -> None:
    rows = []
    for name, tokenizer in tokenizer_dict.items():
        rows.append(
            {
                "name": name,
                "model_input_names": getattr(tokenizer, "model_input_names", None),
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "cls_token": getattr(tokenizer, "cls_token", None),
                "sep_token": getattr(tokenizer, "sep_token", None),
                "unk_token": getattr(tokenizer, "unk_token", None),
                "pad_token": getattr(tokenizer, "pad_token", None),
                "mask_token": getattr(tokenizer, "mask_token", None),
            }
        )
    st.write(DataFrame(rows))


def select_product() -> dict[str, Any]:
    st.write("Select product:")
    df = load_products(locale="jp", nrows=1000)
    df = df[(~df["product_description"].isnull()) & (~df["product_bullet_point"].isnull())]
    df = df.fillna("")

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection("single", use_checkbox=True)
    grid_options = gb.build()
    selected_rows = AgGrid(df, gridOptions=grid_options).selected_rows

    if selected_rows is None or len(selected_rows) <= 0:
        return {}

    return selected_rows.to_dict("records")[0]


@st.fragment
def perform_tokenize(tokenizer_dict: dict[str, PreTrainedTokenizer]):
    product = select_product()
    if not product:
        return

    st.write("### Input")
    s = product["product_title"]
    st.text(s)

    st.write("### Output")
    st.write("`tokenizer.tokenize`")
    rows = []
    for name, tokenizer in tokenizer_dict.items():
        tokens = tokenizer.tokenize(s)
        rows.append(
            {
                "name": name,
                "num_tokens": len(tokens),
                "tokens": tokens,
            }
        )
    st.write(DataFrame(rows))
    st.write("`tokenizer.encode/decode`")
    rows = []
    for name, tokenizer in tokenizer_dict.items():
        token_ids = tokenizer.encode(s)
        rows.append(
            {
                "name": name,
                "num_token_ids": len(token_ids),
                "decoded_token_ids": tokenizer.decode(token_ids),
            }
        )
    st.write(DataFrame(rows))


def main() -> None:
    set_page_config()
    st.write("## AutoTokenizer")

    tokenizer_names = [
        "tohoku-nlp/bert-base-japanese-v3",
        "line-corporation/line-distilbert-base-japanese",
        "hotchpotch/xlm-roberta-japanese-tokenizer",
    ]
    tokenizer_dict = {
        name: get_tokenizer(name)
        for name in tokenizer_names
    }

    draw_tokenizer_info(tokenizer_dict)
    perform_tokenize(tokenizer_dict)


if __name__ == "__main__":
    main()
