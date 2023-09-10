from abc import ABC, abstractmethod
from enum import Enum, auto


class MatchingMethod(Enum):
    SPARSE = auto()
    DENSE = auto()


class WeightingStrategy(ABC):
    @abstractmethod
    def apply(self, matching_method: MatchingMethod, query: str) -> float:
        pass


class FixedWeighting(WeightingStrategy):
    def __init__(self, weight_dict: dict[MatchingMethod, float] | None = None):
        if not weight_dict:
            weight_dict = {
                MatchingMethod.SPARSE: 0.5,
                MatchingMethod.DENSE: 0.5,
            }
        self._weight_dict = weight_dict

    def apply(self, matching_method: MatchingMethod, query: str) -> float:
        return self._weight_dict[matching_method]


class DynamicWeighting(WeightingStrategy):
    def apply(self, matching_method: MatchingMethod, query: str) -> float:
        weight = (len(query) * 0.015) + 0.2
        weight = max(1 - weight, 0.2)
        match matching_method:
            case MatchingMethod.SPARSE:
                return 1 - weight
            case MatchingMethod.DENSE:
                return weight
