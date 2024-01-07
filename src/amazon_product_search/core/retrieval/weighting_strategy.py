from abc import ABC, abstractmethod
from typing import Literal

MatchingMethod = Literal["sparse", "dense"]


class WeightingStrategy(ABC):
    @abstractmethod
    def apply(self, matching_method: MatchingMethod, query: str) -> float:
        pass


class FixedWeighting(WeightingStrategy):
    def __init__(self, weight_dict: dict[MatchingMethod, float] | None = None) -> None:
        if not weight_dict:
            weight_dict = {"sparse": 0.5, "dense": 0.5}
        self._weight_dict = weight_dict

    def apply(self, matching_method: MatchingMethod, query: str) -> float:
        return self._weight_dict[matching_method]
