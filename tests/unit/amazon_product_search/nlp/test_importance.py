import pytest

from amazon_product_search.retrieval.importance_estimator import ColBERTTermImportanceEstimator


@pytest.mark.skip()
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("ナイキの靴", ["ナイキ", "の", "靴"]),
    ],
)
def test_colbert_term_importance_estimator(text, expected):
    estimator = ColBERTTermImportanceEstimator()
    actual = [result[0] for result in estimator.estimate(text)]
    assert actual == expected
