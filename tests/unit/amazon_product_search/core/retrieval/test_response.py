import pytest

from amazon_product_search.core.retrieval.response import Result


@pytest.mark.parametrize(
    ("result", "expected_lexical_score", "expected_semantic_score"),
    [
        (Result(product={}, score=1.0), 0, 0),
        (Result(product={}, score=1.0, explanation={}), 0, 0),
        (Result(product={}, score=1.0, explanation={"lexical_score": 1.0}), 1.0, 0),
        (Result(product={}, score=1.0, explanation={"semantic_score": 1.0}), 0, 1.0),
        (Result(product={}, score=1.0, explanation={"lexical_score": 0.6, "semantic_score": 0.4}), 0.6, 0.4),
    ],
)
def test_get_scores(result, expected_lexical_score, expected_semantic_score):
    assert result.lexical_score == expected_lexical_score
    assert result.semantic_score == expected_semantic_score
