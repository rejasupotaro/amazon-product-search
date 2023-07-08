from typing import Literal

import streamlit as st


def set_page_config(layout: Literal["centered", "wide"] = "wide") -> None:
    st.set_page_config(
        page_title="Search Console",
        page_icon="🛍️",
        layout=layout,
    )
