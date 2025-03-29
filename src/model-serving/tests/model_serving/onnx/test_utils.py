import pytest
import torch
from model_serving.onnx.utils import (
    convert_dict_config_to_dict,
    measure_cosine_similarity,
    measure_mae,
)
from omegaconf import OmegaConf


def test_convert_dict_config_to_dict() -> None:
    config = OmegaConf.create(
        {
            "key1": "value1",
            "key2": [1, 2, 3],
            "key3": {"subkey1": "value1", "subkey2": "value2"},
        }
    )

    actual = convert_dict_config_to_dict(config)
    expected = {
        "key1": "value1",
        "key2": [1, 2, 3],
        "key3": {"subkey1": "value1", "subkey2": "value2"},
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("onnx_embeddings", "torch_embeddings", "expected"),
    [
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.0, 2.0, 3.0]]), 0.0),
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.1, 2.1, 3.1]]), 0.1),
    ],
)
def test_measure_precision(onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor, expected: float) -> None:
    mae = measure_mae(onnx_embeddings, torch_embeddings)
    assert pytest.approx(mae, 0.01) == mae


@pytest.mark.parametrize(
    ("onnx_embeddings", "torch_embeddings", "expected"),
    [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), 1.0),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[-1.0, -1.0]]), -1.0),
    ],
)
def test_measure_cosine_similarity(
    onnx_embeddings: torch.Tensor, torch_embeddings: torch.Tensor, expected: float
) -> None:
    cosine_similarity = measure_cosine_similarity(onnx_embeddings, torch_embeddings)
    assert pytest.approx(cosine_similarity, 0.01) == expected
