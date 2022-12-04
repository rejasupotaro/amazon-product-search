import pandas as pd

from amazon_product_search.synonyms.generator import preprocess_query_title_pairs


def test_preprocess_query_title_pairs():
    df = pd.DataFrame(
        [
            {"query": "Hello", "product_title": "World"},
            {"query": None, "product_title": "product_title"},
            {"query": "query", "product_title": None},
        ]
    )
    df = preprocess_query_title_pairs(df)
    assert df.to_dict("records") == [{"query": "hello", "product_title": "world"}]
