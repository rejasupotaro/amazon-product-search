from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    product: dict[str, Any]
    score: float


@dataclass
class Response:
    results: list[Result]
    total_hits: int
