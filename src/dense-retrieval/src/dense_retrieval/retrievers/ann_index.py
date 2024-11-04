import pickle

import numpy as np
from annoy import AnnoyIndex


class ANNIndex:
    def __init__(self, dim: int) -> None:
        self.annoy_index = AnnoyIndex(dim, "dot")
        self.idx_to_doc_id: dict[int, str] = {}
        self.indexed_doc_ids: set[str] = set()

    def add_items(self, doc_ids: list[str], doc_embs: np.ndarray) -> None:
        for doc_id, doc_emb in zip(doc_ids, doc_embs, strict=True):
            if doc_id in self.indexed_doc_ids:
                continue
            idx = len(self.idx_to_doc_id)
            self.idx_to_doc_id[idx] = doc_id
            self.indexed_doc_ids.add(doc_id)
            self.annoy_index.add_item(idx, doc_emb)

    def build(self) -> None:
        self.annoy_index.build(10)

    def rebuild(self, doc_ids: list[str], doc_embs: np.ndarray) -> None:
        self.add_items(doc_ids, doc_embs)
        self.build()

    def search(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        doc_ids = []
        scores = []
        retrieved = self.annoy_index.get_nns_by_vector(
            query, top_k, include_distances=True
        )
        for idx, score in zip(*retrieved, strict=True):
            doc_ids.append(self.idx_to_doc_id[idx])
            scores.append(score)
        return doc_ids, scores

    def save(self, index_filepath: str) -> None:
        self.annoy_index.save(f"{index_filepath}.ann")
        with open(f"{index_filepath}.pkl", "wb") as file:
            pickle.dump(self.idx_to_doc_id, file)

    def load(self, index_filepath: str) -> None:
        self.annoy_index.load(f"{index_filepath}.ann")
        with open(f"{index_filepath}.pkl", "rb") as file:
            self.idx_to_doc_id = pickle.load(file)
