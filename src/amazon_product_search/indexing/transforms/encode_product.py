import logging
from typing import Any, Dict, Iterator, List, Tuple

import apache_beam as beam
from apache_beam.utils.shared import Shared
from torch import Tensor
from tritonclient.grpc import (
    InferenceServerClient,
    InferInput,
)

from amazon_product_search.indexing.transforms.weak_reference import WeakReference
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder
from amazon_product_search_dense_retrieval.encoders.modules.pooler import PoolingMode


def _product_to_text(product: Dict[str, Any], fields: list[str]) -> str:
    return " ".join(product[field] for field in fields)


class EncodeProductFn(beam.DoFn):
    def __init__(
        self,
        shared_handle: Shared,
        hf_model_name: str,
        product_fields: list[str],
        pooling_mode: PoolingMode = "mean",
    ) -> None:
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._pooling_mode = pooling_mode
        self._product_fields = product_fields

    def setup(self) -> None:
        def initialize_encoder() -> WeakReference[SBERTEncoder]:
            # Load a potentially large model in memory. Executed once per process.
            return WeakReference[SBERTEncoder](SBERTEncoder(self._hf_model_name, self._pooling_mode))

        self._weak_reference: WeakReference[SBERTEncoder] = self._shared_handle.acquire(initialize_encoder)

    def _encode(self, texts: list[str]) -> Tensor:
        return self._weak_reference.ref.encode(texts)

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, List[float]]]:
        logging.info(f"Encode {len(products)} products in a batch")
        texts = [_product_to_text(product, self._product_fields) for product in products]
        product_vectors = self._encode(texts)
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector.tolist()


class EncodeProductTritonFn(beam.DoFn):
    def __init__(
        self,
        hf_model_name: str,
        product_fields: list[str],
        host: str = "localhost:8001",
        onnx_model_name: str = "text_embedding",
    ) -> None:
        self._hf_model_name = hf_model_name
        self._product_fields = product_fields
        self._host = host
        self._onnx_model_name = onnx_model_name

    def setup(self) -> None:
        self._client = InferenceServerClient(host=self._host)

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, List[float]]]:
        texts = [_product_to_text(product, self._product_fields) for product in products]
        product_vectors = self._client.infer(
            model_name=self._onnx_model_name,
            inputs=[
                InferInput(
                    name="text",
                    shape=[len(texts)],
                    datatype="BYTES",
                )
            ],
        ).as_numpy("output")
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector.tolist()


class EncodeProduct(beam.PTransform):
    def __init__(
        self,
        shared_handle: Shared,
        hf_model_name: str,
        batch_size: int,
        product_fields: list[str] | None = None,
        use_triton: bool = False,
    ) -> None:
        super().__init__()
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._batch_size = batch_size
        if not product_fields:
            product_fields = ["product_title"]
        self._product_fields = product_fields
        self._use_triton = use_triton

    def expand(self, pcoll: beam.PCollection[Dict[str, Any]]) -> beam.PCollection[Dict[str, Any]]:
        return (
            (pcoll | "Batch items for EncodeProductFn" >> beam.BatchElements(min_batch_size=self._batch_size))
            | (
                beam.ParDo(
                    EncodeProductFn(
                        shared_handle=self._shared_handle,
                        hf_model_name=self._hf_model_name,
                        product_fields=self._product_fields,
                    )
                )
            )
            if not self._use_triton
            else (
                beam.ParDo(
                    EncodeProductTritonFn(
                        hf_model_name=self._hf_model_name,
                        product_fields=self._product_fields,
                    )
                )
            )
        )
