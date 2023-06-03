import requests
import streamlit as st

from amazon_product_search.vespa.vespa_client import VespaClient

client = VespaClient()


def main() -> None:
    endpoints = [
        "http://localhost:19071/ApplicationStatus",
        "http://localhost:8080/document/v1/amazon/product/docid",
    ]
    for endpoint in endpoints:
        res = requests.get(endpoint)
        with st.expander(f"GET {endpoint}", expanded=False):
            st.write(res.json())

    st.write("# Search")

    query = st.text_input("Query:")
    vespa_query = {
        "yql": "select * from sources * where userQuery()",
        "query": query,
        "type": "any",
        "ranking": "random",
        "hits": 10,
    }
    st.write(vespa_query)

    if not st.button("Search"):
        return

    response = client.search(vespa_query)
    st.write(response.json)


if __name__ == "__main__":
    main()
