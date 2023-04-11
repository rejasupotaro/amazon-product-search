from amazon_product_search.synonyms.filters.similarity_filter import SimilarityFilter


def test_calculate_score():
    left = ["text", "text"]
    right = ["text", "text"]

    filter = SimilarityFilter()
    scores = filter.calculate_score(left, right)

    assert len(scores) == 2
