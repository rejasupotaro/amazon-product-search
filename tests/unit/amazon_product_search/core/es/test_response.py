import pytest

from amazon_product_search.core.es.response import Result


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
