from typing import Any

import numpy as np
import onnx
import torch
from omegaconf import DictConfig
from onnxruntime import InferenceSession
from transformers import BatchEncoding


def convert_dict_config_to_dict(d: DictConfig | Any) -> Any:
    """Convert a DictConfig object to a regular dictionary.

    This is necessary because DictConfig objects are not directly
    compatible with `torch.onnx.export` that expect a standard dict.
    """
    if isinstance(d, DictConfig):
        return {k: convert_dict_config_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_dict_config_to_dict(item) for item in d]
    else:
        return d


def print_input_and_output_names(onnx_model_filepath: str) -> None:
    model = onnx.load(onnx_model_filepath)
    for i, input in enumerate(model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    for i, output in enumerate(model.graph.output):
        print("[Output #{}]".format(i))
        print(output)


def generate_embeddings(
    onnx_params: dict[str, Any], onnx_model_filepath: str, tokenized: BatchEncoding
) -> torch.Tensor:
    session = InferenceSession(onnx_model_filepath)
    last_hidden_state = session.run(
        output_names=onnx_params["output_names"],
        input_feed={input_name: tokenized[input_name].numpy() for input_name in onnx_params["input_names"]},
    )[0]

    attention_mask = tokenized["attention_mask"]
    masked_hidden_state = last_hidden_state * np.expand_dims(attention_mask, axis=-1)
    embeddings = masked_hidden_state.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
    return embeddings


def measure_mae(onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor) -> torch.Tensor:
    """Measure the precision between ONNX and PyTorch embeddings using Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(onnx_embeddings - torch_embeddings))


def measure_cosine_similarity(onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor) -> torch.Tensor:
    """Measure the cosine similarity between ONNX and PyTorch embeddings."""
    dot_products = torch.sum(onnx_embeddings * torch_embeddings, dim=1)
    norms = torch.norm(onnx_embeddings, dim=1) * torch.norm(torch_embeddings, dim=1)
    cosine_similarities = dot_products / norms
    return torch.mean(cosine_similarities)


def verify_onnx_model(
    onnx_params: dict[str, Any], onnx_model_filepath: str, tokenized: BatchEncoding, torch_embeddings: torch.Tensor
) -> None:
    print(f"Verifying {onnx_model_filepath}...")
    model = onnx.load(onnx_model_filepath)
    onnx.checker.check_model(model)
    print("ONNX model is valid")

    onnx_embeddings = generate_embeddings(onnx_params, onnx_model_filepath, tokenized)

    # Measure precision using Mean Absolute Error (MAE)
    mae = measure_mae(onnx_embeddings, torch_embeddings)
    print(f"Mean Absolute Error (MAE) between ONNX and PyTorch embeddings: {mae:.8f}")

    # Optionally, measure cosine similarity
    cosine_similarity = measure_cosine_similarity(onnx_embeddings, torch_embeddings)
    print(f"Cosine Similarity between ONNX and PyTorch embeddings: {cosine_similarity:.8f}")
