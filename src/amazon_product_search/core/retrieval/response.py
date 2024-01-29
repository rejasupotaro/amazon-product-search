from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Result:
    product: dict[str, Any]
    score: float
    explanation: Optional[dict[str, Any]] = None

    @property
    def lexical_score(self) -> float:
        if not self.explanation:
            return 0
        return self.explanation.get("lexical_score", 0) if self.explanation else 0

    @property
    def semantic_score(self) -> float:
        if not self.explanation:
            return 0
        return self.explanation.get("semantic_score", 0) if self.explanation else 0

    @staticmethod
    def _get_semantic_score(explanation: dict[str, Any]) -> float:
        """Attempts to extract the semantic score from an explanation.

        This function assumes that the explanation is retrieved from Elasticsearch.

        Args:
            explanation (dict[str, Any]): An explanation object.

        Returns:
            float: The semantic score.
        """
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

        if self.lexical_score > 0 or self.semantic_score > 0:
            return self.lexical_score, self.semantic_score

        dense_score = self._get_semantic_score(self.explanation)
        if dense_score:
            return 0, dense_score

        sparse_score, dense_score = self.score, 0
        for child_explanation in self.explanation.get("details", []):
            dense_score += self._get_semantic_score(child_explanation)
        return sparse_score - dense_score, dense_score


@dataclass
class Response:
    results: list[Result]
    total_hits: int
