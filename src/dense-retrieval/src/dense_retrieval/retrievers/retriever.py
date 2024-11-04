from typing import Protocol

import numpy as np


class Retriever(Protocol):
    """A retriever that finds the most relevant documents to a given query.

    This class is just for training and evaluation. In practice, you should use a distributed search engine
    that is capable to handle large-scale documents.
    """

    def retrieve(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        ...
