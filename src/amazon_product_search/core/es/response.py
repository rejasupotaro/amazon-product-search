from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Result:
    product: dict[str, Any]
    score: float
    explanation: Optional[dict[str, Any]] = None

    @staticmethod
    def _get_dense_score(explanation: dict[str, Any]) -> float:
        if explanation.get("description") != "within top k documents":
            return 0
        return explanation.get("value", 0)

    def get_scores_in_explanation(self) -> tuple[float, float]:
        """Get a summary of scoring.

        If the result has no explanation, (0, 0) is returned.

        Returns:
            tuple[float, float]: A tuple of (sparse_score, dense_score).
        """
        if not self.explanation:
            return 0, 0

        sparse_score, dense_score = self.explanation.get("sparse_score"), self.explanation.get("dense_score")
        if sparse_score is not None and dense_score is not None:
            return sparse_score, dense_score

        dense_score = self._get_dense_score(self.explanation)
        if dense_score:
            return 0, dense_score

        sparse_score, dense_score = self.score, 0
        for child_explanation in self.explanation.get("details", []):
            dense_score += self._get_dense_score(child_explanation)
        return sparse_score - dense_score, dense_score


@dataclass
class Response:
    results: list[Result]
    total_hits: int
