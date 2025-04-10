import logging
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple

import apache_beam as beam
from apache_beam.utils.shared import Shared

from dense_retrieval.encoders import SBERTEncoder
from dense_retrieval.encoders.modules.pooler import PoolingMode


def _product_to_text(product: Dict[str, Any], fields: list[str]) -> str:
    return " ".join(product[field] for field in fields)


def initialize_encoder(hf_model_name: str, pooling_mode: PoolingMode) -> SBERTEncoder:
    return SBERTEncoder(hf_model_name, pooling_mode)


class EncodeProductFn(beam.DoFn):
    def __init__(
        self,
        shared_handle: Shared,
        hf_model_name: str,
        product_fields: list[str],
        pooling_mode: PoolingMode = "mean",
    ) -> None:
        super().__init__()
        self._shared_handle = shared_handle
        self._initialize_fn = partial(initialize_encoder, hf_model_name, pooling_mode)
        self._product_fields = product_fields

    def setup(self) -> None:
        self._encoder: SBERTEncoder = self._shared_handle.acquire(self._initialize_fn)

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, List[float]]]:
        logging.info(f"Encode {len(products)} products in a batch")
        texts = [_product_to_text(product, self._product_fields) for product in products]
        product_vectors = self._encoder.encode(texts)
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector.tolist()


class EncodeProduct(beam.PTransform):
    def __init__(
        self,
        shared_handle: Shared,
        hf_model_name: str,
        batch_size: int,
        product_fields: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._batch_size = batch_size
        if not product_fields:
            product_fields = ["product_title"]
        self._product_fields = product_fields

    def expand(self, pcoll: beam.PCollection[Dict[str, Any]]) -> beam.PCollection[Tuple[str, List[float]]]:
        pcoll |= "Batch items for EncodeProductFn" >> beam.BatchElements(min_batch_size=self._batch_size)
        return pcoll | beam.ParDo(
            EncodeProductFn(
                shared_handle=self._shared_handle,
                hf_model_name=self._hf_model_name,
                product_fields=self._product_fields,
            )
        )
