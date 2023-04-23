from typing import List, Protocol

from torch import Tensor


class Encoder(Protocol):
    def encode(self, texts: str | List[str]) -> Tensor:
        ...
