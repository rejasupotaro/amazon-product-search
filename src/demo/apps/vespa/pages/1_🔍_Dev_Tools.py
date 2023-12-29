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


if __name__ == "__main__":
    main()
