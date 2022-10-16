from dataclasses import dataclass
from typing import Any


@dataclass
class RequestParams:
    query: str
    use_description: bool = False
    use_bullet_point: bool = False
    use_brand: bool = False
    use_color_name: bool = False


@dataclass
class Result:
    product: dict[str, Any]
    score: float


@dataclass
class Response:
    results: list[Result]
    total_hits: int
