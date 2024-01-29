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


@pytest.mark.parametrize(
    ("explanation", "expected"),
    [
        (None, (0, 0)),
        ({}, (0, 0)),
        (
            {
                "value": 1,
                "description": "result of:",
                "details": [],
            },
            (1, 0),
        ),
        (
            {
                "value": 1,
                "description": "within top k documents",
                "details": [],
            },
            (0, 1),
        ),
        (
            {
                "value": 1,
                "description": "sum of:",
                "details": [
                    {
                        "value": 0.1,
                        "description": "within top k documents",
                        "details": [],
                    },
                ],
            },
            (0.9, 0.1),
        ),
    ],
)
def test_get_scores_in_explanation(explanation, expected):
    result = Result(
        product={},
        score=1,
        explanation=explanation,
    )
    assert result.get_scores_in_explanation() == expected
