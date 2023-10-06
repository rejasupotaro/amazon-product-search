import json

import requests
import streamlit as st

from amazon_product_search.core.vespa.vespa_client import VespaClient

client = VespaClient()


def main() -> None:
    endpoints = [
        "http://localhost:8080/state/v1",
        "http://localhost:8080/metrics/v2/values",
        "http://localhost:19071/ApplicationStatus",
        "http://localhost:8080/document/v1/amazon/product/docid",
    ]
    for endpoint in endpoints:
        res = requests.get(endpoint)
        with st.expander(f"GET {endpoint}", expanded=False):
            st.write(res.json())

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
