from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Result:
    product: dict[str, Any]
    score: float
    explanation: Optional[dict[str, Any]] = None


@dataclass
class Response:
    results: list[Result]
    total_hits: int
