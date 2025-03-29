import os

import torch
from omegaconf import DictConfig
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from model_serving.onnx.utils import (
    convert_dict_config_to_dict,
    verify_onnx_model,
)
from transformers import AutoModel, AutoTokenizer


def export(export_params: DictConfig) -> None:
    """Export a model to ONNX format.

    The configuration is located in `configs/pipeline/export_parameters/*`.
    Based on it, an ONNX model is generated and saved to the `models` directory.

    Args:
        export_parameters (DictConfig): A configuration object containing the parameters for exporting the model.
    """
    # The model name is expected to be in the format "org_name/model_name",
    # for example, "hotchpotch/static-embedding-japanese".
    model_full_name = export_params.model_name  # "org_name/model_name"
    model_org_name, model_name = export_params.model_name.split("/")  # "org_name", "model_name"
    onnx_model_dir = f"models/{model_org_name}"  # "models/org_name"
    onnx_model_filepath = f"{onnx_model_dir}/{model_name}.onnx"  # "models/org_name/model_name.onnx"
    preprocessed_model_filepath = (
        f"{onnx_model_dir}/{model_name}_preprocessed.onnx"  # "models/org_name/model_name_preprocessed.onnx"
    )
    quantized_onnx_model_filepath = (
        f"{onnx_model_dir}/{model_name}_quantized.onnx"  # "models/org_name/model_name_quantized.onnx"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_full_name, trust_remote_code=True)
    tokenized = tokenizer(
        "dummy_input",
        return_tensors="pt",
        **export_params.tokenizer_parameters,
    )

    model = AutoModel.from_pretrained(model_full_name)
    model.eval()

    # Generate embeddings (torch.Tensor) to verify the model output.
    model_output = model(**tokenized)
    last_hidden_state = model_output.last_hidden_state
    attention_mask = tokenized["attention_mask"]
    masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
    embeddings = masked_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    # Create the directory if it does not exist.
    os.makedirs(onnx_model_dir, exist_ok=True)

    print("Exporting to ONNX format...")
    onnx_params = convert_dict_config_to_dict(export_params.onnx_parameters)
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
