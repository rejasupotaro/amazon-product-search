import logging
from typing import Any, Dict, Iterator, List, Tuple

import apache_beam as beam
import numpy as np
from apache_beam.utils.shared import Shared
from torch import Tensor
from transformers import AutoTokenizer
from tritonclient.grpc import (
    InferenceServerClient,
    InferInput,
)

from amazon_product_search.indexing.transforms.weak_reference import WeakReference
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder
from amazon_product_search_dense_retrieval.encoders.modules.pooler import PoolingMode


class EncodeProductFn(beam.DoFn):
    def __init__(self, shared_handle: Shared, hf_model_name: str, pooling_mode: PoolingMode = "mean") -> None:
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._pooling_mode = pooling_mode

    def setup(self) -> None:
        def initialize_encoder() -> WeakReference[SBERTEncoder]:
            # Load a potentially large model in memory. Executed once per process.
            return WeakReference[SBERTEncoder](SBERTEncoder(self._hf_model_name, self._pooling_mode))

        self._weak_reference: WeakReference[SBERTEncoder] = self._shared_handle.acquire(initialize_encoder)

    def _encode(self, texts: list[str]) -> Tensor:
        return self._weak_reference.ref.encode(texts)

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, List[float]]]:
        logging.info(f"Encode {len(products)} products in a batch")
        texts = [product["product_title"] + " " + product["product_brand"] for product in products]
        product_vectors = self._encode(texts)
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector.tolist()


class EncodeProductTritonFn(beam.DoFn):
    def __init__(self, hf_model_name: str, host: str = "localhost:8001", onnx_model_name: str = "all_minilm") -> None:
        self._hf_model_name = hf_model_name
        self._host = host
        self._onnx_model_name = onnx_model_name

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name)
        self._client = InferenceServerClient(host=self._host)

    def _tokenize(self, texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        return self._tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

    def process(self, products: List[Dict[str, Any]]) -> Iterator[Tuple[str, List[float]]]:
        input_ids, attention_mask = self._tokenize(
            [product["product_title"] + " " + product["product_brand"] for product in products]
        )
        product_vectors = self._client.infer(
            model_name=self._onnx_model_name,
            inputs=[
                InferInput(
                    name="input_ids",
                    shape=[len(products), 512],
                    datatype="INT64",
                ).set_data_from_numpy(input_ids),
                InferInput(
                    name="attention_mask",
                    shape=[len(products), 512],
                    datatype="INT64",
                ).set_data_from_numpy(attention_mask),
            ],
        ).as_numpy("output")
        for product, product_vector in zip(products, product_vectors, strict=True):
            yield product["product_id"], product_vector.tolist()


class EncodeProduct(beam.PTransform):
    def __init__(self, shared_handle: Shared, hf_model_name: str, batch_size: int, use_triton: bool = False) -> None:
        super().__init__()
        self._shared_handle = shared_handle
        self._hf_model_name = hf_model_name
        self._batch_size = batch_size
        self._use_triton = use_triton

    def expand(self, pcoll: beam.PCollection[Dict[str, Any]]) -> beam.PCollection[Dict[str, Any]]:
        return (
            (pcoll | "Batch items for EncodeProductFn" >> beam.BatchElements(min_batch_size=self._batch_size))
            | (
                beam.ParDo(
                    EncodeProductFn(
                        shared_handle=self._shared_handle,
                        hf_model_name=self._hf_model_name,
                    )
                )
            )
            if not self._use_triton
            else (
                beam.ParDo(
                    EncodeProductTritonFn(
                        hf_model_name=self._hf_model_name,
                    )
                )
            )
        )
