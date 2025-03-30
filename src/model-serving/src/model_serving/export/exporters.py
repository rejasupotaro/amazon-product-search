import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import onnx
import torch
from model_serving.export.utils import (
    convert_dict_config_to_dict,
    measure_cosine_similarity,
    measure_mae,
    print_input_and_output_names,
)
from model_serving.modules.sentence_transformer_wrapper import SentenceTransformerWrapper
from omegaconf import DictConfig
from onnxruntime import InferenceSession
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import BatchEncoding

logger = logging.getLogger(__name__)


class BaseModelExporter(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # The model name is expected to be in the format "org_name/model_name",
        # for example, "hotchpotch/static-embedding-japanese".
        self.model_full_name = cfg.export_params.model_name  # "org_name/model_name"
        self.model_org_name, self.model_name = self.cfg.export_params.model_name.split("/")  # "org_name", "model_name"
        self.model_dir = f"models/{self.model_org_name}"  # "models/org_name"

        self.original_model = SentenceTransformerWrapper(self.model_full_name)
        self.original_model.eval()

        # Create the directory if it does not exist.
        os.makedirs(self.model_dir, exist_ok=True)

        self.tokenized = self.tokenize_dummy_input()

    def get_filepath(self, suffix: str) -> str:
        return f"{self.model_dir}/{self.model_name}{suffix}"

    def tokenize_dummy_input(self) -> dict[str, torch.Tensor]:
        return self.original_model.tokenizer(
            "dummy_input",
            return_tensors="pt",
            **self.cfg.export_params.tokenizer_parameters,
        )

    def compare_embeddings(
        self,
        exported_embeddings: torch.Tensor,
        original_embeddings: torch.Tensor,
    ) -> None:
        # Measure precision using Mean Absolute Error (MAE)
        mae = measure_mae(exported_embeddings, original_embeddings)
        logger.info(f"Mean Absolute Error (MAE): {mae:.8f}")

        # Optionally, measure cosine similarity
        cosine_similarity = measure_cosine_similarity(exported_embeddings, original_embeddings)
        logger.info(f"Cosine Similarity: {cosine_similarity:.8f}")


    @abstractmethod
    def export(self) -> None:
        raise NotImplementedError


class TorchScriptModelExporter(BaseModelExporter):
    def export(self) -> None:
        model_filepath = self.get_filepath(".pt")

        # Generate embeddings (torch.Tensor) to verify the model output.
        original_embeddings = self.original_model(**self.tokenized)

        logger.info("Exporting to TorchScript format...")
        # Use torch.jit.trace for static models
        traced_script_module = torch.jit.trace(
            func=self.original_model,
            example_inputs=tuple(self.tokenized.values()),
        )
        torch.jit.save(traced_script_module, model_filepath)

        logger.info(f"Verifying {model_filepath}...")
        exported_model = torch.jit.load(model_filepath)
        exported_model.eval()

        exported_embeddings = exported_model(**self.tokenized)
        self.compare_embeddings(exported_embeddings, original_embeddings)
        logger.info(f"TorchScript model saved to {model_filepath}")


class ONNXModelExporter(BaseModelExporter):
    def generate_embeddings(
        self, onnx_params: dict[str, Any], model_filepath: str, tokenized: BatchEncoding,
    ) -> torch.Tensor:
        session = InferenceSession(model_filepath)
        embeddings = session.run(
            output_names=onnx_params["output_names"],
            input_feed={input_name: tokenized[input_name].numpy() for input_name in onnx_params["input_names"]},
        )[0]
        return torch.tensor(embeddings)

    def verify_model(
        self,
        onnx_params: dict[str, Any],
        model_filepath: str,
        tokenized: BatchEncoding,
        original_embeddings: torch.Tensor,
    ) -> None:
        logger.info(f"Verifying {model_filepath}...")
        exported_model = onnx.load(model_filepath)
        onnx.checker.check_model(exported_model)
        logger.info("ONNX model is valid")

        exported_embeddings = self.generate_embeddings(onnx_params, model_filepath, tokenized)
        self.compare_embeddings(exported_embeddings, original_embeddings)

    def export(self) -> None:
        model_filepath = self.get_filepath(".onnx")
        preprocessed_model_filepath = self.get_filepath("_preprocessed.onnx")
        quantized_model_filepath = self.get_filepath("_quantized.onnx")

        # Generate embeddings (torch.Tensor) to verify the model output.
        tokenized = self.tokenize_dummy_input()
        original_embeddings = self.original_model(**tokenized)

        logger.info("Exporting to ONNX format...")
        onnx_params = convert_dict_config_to_dict(self.cfg.export_params.onnx_parameters)
        # For available options, see: https://glaringlee.github.io/onnx.html#functions
        torch.onnx.export(
            self.original_model,
            args=tuple(tokenized.values()),
            f=model_filepath,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            **onnx_params,
        )
        self.verify_model(onnx_params, model_filepath, tokenized, original_embeddings)
        logger.info(f"ONNX model saved to {model_filepath}")

        logger.info("Preprocessing ONNX model for quantization...")
        quant_pre_process(
            model_filepath,
            preprocessed_model_filepath,
            auto_merge=True,
        )
        self.verify_model(onnx_params, preprocessed_model_filepath, tokenized, original_embeddings)
        logger.info(f"Preprocessed ONNX model saved to {preprocessed_model_filepath}")

        quantize_dynamic(preprocessed_model_filepath, quantized_model_filepath)
        self.verify_model(onnx_params, quantized_model_filepath, tokenized, original_embeddings)
        logger.info(f"Quantized ONNX model saved to {quantized_model_filepath}")

        if self.cfg.verbose:
            print_input_and_output_names(quantized_model_filepath)
