from dataclasses import dataclass
from typing import Literal


@dataclass
class RankFusion:
    fuser: Literal["search_engine", "own"] = "search_engine"
    fusion_strategy: Literal["fuse", "append"] = "fuse"
    # When fusion_strategy is "fuse", the following options are available.
    enable_append_dense: bool = False
    enable_score_normalization: bool = False
    rrf: bool | int = False
    weighting_strategy: Literal["fixed", "dynamic"] = "fixed"
