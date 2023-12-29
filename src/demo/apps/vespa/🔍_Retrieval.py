import json

import streamlit as st

from amazon_product_search.core.vespa.vespa_client import VespaClient

client = VespaClient()


def main() -> None:
    st.write("# Search")

    query_str = """
{
    "yql": "select * from product where userQuery()",
    "query": "query",
    "type": "any",
    "ranking": "random",
    "hits": 10
}
    """.strip()
    query_str = st.text_area("Query:", value=query_str, height=300)
    query = json.loads(query_str)

    if not st.button("Search"):
        return

    response = client.search(query)
    st.write(response.json)


if __name__ == "__main__":
    main()
