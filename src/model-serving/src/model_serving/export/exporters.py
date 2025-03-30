import os
from abc import ABC, abstractmethod

import torch
from model_serving.export.utils import (
    convert_dict_config_to_dict,
    print_input_and_output_names,
    verify_onnx_model,
    verify_torchscript_model,
)
from model_serving.modules.sentence_transformer_wrapper import SentenceTransformerWrapper
from omegaconf import DictConfig
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


class BaseModelExporter(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # The model name is expected to be in the format "org_name/model_name",
        # for example, "hotchpotch/static-embedding-japanese".
        self.model_full_name = cfg.export_params.model_name  # "org_name/model_name"
        self.model_org_name, self.model_name = self.cfg.export_params.model_name.split("/")  # "org_name", "model_name"
        self.model_dir = f"models/{self.model_org_name}"  # "models/org_name"

        self.model = SentenceTransformerWrapper(self.model_full_name)
        self.model.eval()

        # Create the directory if it does not exist.
        os.makedirs(self.model_dir, exist_ok=True)

    def tokenize_dummy_input(self) -> dict[str, torch.Tensor]:
        return self.model.tokenizer(
            "dummy_input",
            return_tensors="pt",
            **self.cfg.export_params.tokenizer_parameters,
        )

    @abstractmethod
    def export(self) -> None:
        raise NotImplementedError


class TorchScriptModelExporter(BaseModelExporter):
    @property
    def model_filepath(self) -> str:
        return f"{self.model_dir}/{self.model_name}.pt"  # "models/org_name/model_name.pt"

    def export(self) -> None:
        # Generate embeddings (torch.Tensor) to verify the model output.
        tokenized = self.tokenize_dummy_input()
        embeddings = self.model(**tokenized)

        print("Exporting to TorchScript format...")
        # Use torch.jit.trace for static models
        traced_script_module = torch.jit.trace(
            func=self.model,
            example_inputs=tuple(tokenized.values()),
        )
        torch.jit.save(traced_script_module, self.model_filepath)
        verify_torchscript_model(self.model_filepath, tokenized, embeddings)
        print(f"TorchScript model saved to {self.model_filepath}")


class ONNXModelExporter(BaseModelExporter):
    @property
    def model_filepath(self) -> str:
        return f"{self.model_dir}/{self.model_name}.onnx"  # "models/org_name/model_name.onnx"

    @property
    def preprocessed_model_filepath(self) -> str:
        return f"{self.model_dir}/{self.model_name}_preprocessed.onnx"  # "models/org_name/model_name_preprocessed.onnx"

    @property
    def quantized_model_filepath(self) -> str:
        return f"{self.model_dir}/{self.model_name}_quantized.onnx"  # "models/org_name/model_name_quantized.onnx"
    def export(self) -> None:
        # Generate embeddings (torch.Tensor) to verify the model output.
        tokenized = self.tokenize_dummy_input()
        embeddings = self.model(**tokenized)

        print("Exporting to ONNX format...")
        onnx_params = convert_dict_config_to_dict(self.cfg.export_params.onnx_parameters)
        # For available options, see: https://glaringlee.github.io/onnx.html#functions
        torch.onnx.export(
            self.model,
            args=tuple(tokenized.values()),
            f=self.model_filepath,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            **onnx_params,
        )
        verify_onnx_model(onnx_params, self.model_filepath, tokenized, embeddings)
        print(f"ONNX model saved to {self.model_filepath}")

        print("Preprocessing ONNX model for quantization...")
        quant_pre_process(
            self.model_filepath,
            self.preprocessed_model_filepath,
            auto_merge=True,
        )
        verify_onnx_model(onnx_params, self.preprocessed_model_filepath, tokenized, embeddings)
        print(f"Preprocessed ONNX model saved to {self.preprocessed_model_filepath}")

        quantize_dynamic(self.preprocessed_model_filepath, self.quantized_model_filepath)
        verify_onnx_model(onnx_params, self.quantized_model_filepath, tokenized, embeddings)
        print(f"Quantized ONNX model saved to {self.quantized_model_filepath}")

        if self.cfg.verbose:
            print_input_and_output_names(self.quantized_model_filepath)
