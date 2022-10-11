from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RequestParams:
    query: str
    use_description: bool


@dataclass
class Result:
    product: Dict[str, Any]
    score: float


@dataclass
class Response:
    results: List[Result]
    total_hits: int
