import os

import torch
from model_serving.modules.sentence_transformer_wrapper import SentenceTransformerWrapper
from model_serving.onnx.utils import (
    convert_dict_config_to_dict,
    print_input_and_output_names,
    verify_onnx_model,
)
from omegaconf import DictConfig
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def export(cfg: DictConfig) -> None:
    """Export a model to ONNX format.

    The configuration is located in `configs/pipeline/export_parameters/*`.
    Based on it, an ONNX model is generated and saved to the `models` directory.
    """
    export_params = cfg.export_params
    # The model name is expected to be in the format "org_name/model_name",
    # for example, "hotchpotch/static-embedding-japanese".
    model_full_name = export_params.model_name  # "org_name/model_name"
    model_org_name, model_name = export_params.model_name.split("/")  # "org_name", "model_name"
    model_dir = f"models/{model_org_name}"  # "models/org_name"

    model = SentenceTransformerWrapper(model_full_name)
    model.eval()

    # Create the directory if it does not exist.
    os.makedirs(model_dir, exist_ok=True)

    export_model_to_torchscript(cfg, model, model_dir, model_name)
    export_model_to_onnx(cfg, model, model_dir, model_name)


def export_model_to_torchscript(
    cfg: DictConfig,
    model: SentenceTransformerWrapper,
    model_dir: str,
    model_name: str,
) -> None:
    torchscript_model_filepath = f"{model_dir}/{model_name}.pt"  # "models/org_name/model_name.onnx"

    tokenized = model.tokenizer(
        "dummy_input",
        return_tensors="pt",
        **cfg.export_params.tokenizer_parameters,
    )

    print("Exporting to TorchScript format...")
    # Use torch.jit.trace for static models
    traced_script_module = torch.jit.trace(
        func=model,
        example_inputs=tuple(tokenized.values()),
    )
    torch.jit.save(traced_script_module, torchscript_model_filepath)
    print(f"TorchScript model saved to {torchscript_model_filepath}")


def export_model_to_onnx(
    cfg: DictConfig,
    model: SentenceTransformerWrapper,
    model_dir: str,
    model_name: str,
) -> None:
    onnx_model_filepath = f"{model_dir}/{model_name}.onnx"  # "models/org_name/model_name.onnx"
    preprocessed_model_filepath = (
        f"{model_dir}/{model_name}_preprocessed.onnx"  # "models/org_name/model_name_preprocessed.onnx"
    )
    quantized_onnx_model_filepath = (
        f"{model_dir}/{model_name}_quantized.onnx"  # "models/org_name/model_name_quantized.onnx"
    )

    tokenized = model.tokenizer(
        "dummy_input",
        return_tensors="pt",
        **cfg.export_params.tokenizer_parameters,
    )

    # Generate embeddings (torch.Tensor) to verify the model output.
    embeddings = model(**tokenized)

    print("Exporting to ONNX format...")
    onnx_params = convert_dict_config_to_dict(cfg.export_params.onnx_parameters)
    # For available options, see: https://glaringlee.github.io/onnx.html#functions
    torch.onnx.export(
        model,
        args=tuple(tokenized.values()),
        f=onnx_model_filepath,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        **onnx_params,
    )
    verify_onnx_model(onnx_params, onnx_model_filepath, tokenized, embeddings)
    print(f"ONNX model saved to {onnx_model_filepath}")

    print("Preprocessing ONNX model for quantization...")
    quant_pre_process(
        onnx_model_filepath,
        preprocessed_model_filepath,
        auto_merge=True,
    )
    verify_onnx_model(onnx_params, preprocessed_model_filepath, tokenized, embeddings)
    print(f"Preprocessed ONNX model saved to {preprocessed_model_filepath}")

    quantize_dynamic(preprocessed_model_filepath, quantized_onnx_model_filepath)
    verify_onnx_model(onnx_params, quantized_onnx_model_filepath, tokenized, embeddings)
    print(f"Quantized ONNX model saved to {quantized_onnx_model_filepath}")

    if cfg.verbose:
        print_input_and_output_names(quantized_onnx_model_filepath)
