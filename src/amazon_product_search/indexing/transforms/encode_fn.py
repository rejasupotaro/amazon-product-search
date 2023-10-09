import logging
from typing import Any, Dict, Iterator, List, Tuple

import apache_beam as beam
from apache_beam.utils.shared import Shared
from torch import Tensor

from amazon_product_search.indexing.transforms.weak_reference import WeakReference
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


class EncodeFn(beam.DoFn):
    def __init__(self, hf_model_name: str) -> None:
        self._hf_model_name = hf_model_name

    def setup(self) -> None:
        self.encoder: SBERTEncoder = SBERTEncoder(self._hf_model_name)

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        text = product["product_title"] + " " + product["product_brand"]
        product_vectors = self.encoder.encode([text])
        product["product_vector"] = product_vectors[0]
        yield product


class BatchEncodeFn(beam.DoFn):
    def __init__(self, shared_handle: Shared, hf_model_name: str) -> None:
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name

    def setup(self) -> None:
        def initialize_encoder() -> WeakReference[SBERTEncoder]:
            # Load a potentially large model in memory. Executed once per process.
            return WeakReference[SBERTEncoder](SBERTEncoder(self._hf_model_name))

        self._weak_reference: WeakReference[SBERTEncoder] = self._shared_handle.acquire(initialize_encoder)

    def _encode(self, texts: list[str]) -> Tensor:
        return self._weak_reference.ref.encode(texts)

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, Tensor]]:
        logging.info(f"Encode {len(products)} products in a batch")
        texts = [product["product_title"] + " " + product["product_brand"] for product in products]
        product_vectors = self._encode(texts)
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector
