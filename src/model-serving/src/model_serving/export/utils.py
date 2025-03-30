from typing import Any

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
    embeddings = session.run(
        output_names=onnx_params["output_names"],
        input_feed={input_name: tokenized[input_name].numpy() for input_name in onnx_params["input_names"]},
    )[0]
    return torch.tensor(embeddings)


def measure_mae(onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor) -> torch.Tensor:
    """Measure the precision between ONNX and PyTorch embeddings using Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(onnx_embeddings - torch_embeddings))


def measure_cosine_similarity(onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor) -> torch.Tensor:
    """Measure the cosine similarity between ONNX and PyTorch embeddings."""
    dot_products = torch.sum(onnx_embeddings * torch_embeddings, dim=1)
    norms = torch.norm(onnx_embeddings, dim=1) * torch.norm(torch_embeddings, dim=1)
    cosine_similarities = dot_products / norms
    return torch.mean(cosine_similarities)
