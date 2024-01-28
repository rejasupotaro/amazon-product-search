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


@dataclass
class Response:
    results: list[Result]
    total_hits: int
