from typing import Any

import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args: dict[str, Any]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("cl-nagoya/ruri-small-v2", trust_remote_code=True)

    def _tokenize(self, texts: list[str]) -> dict[str, Any]:
        return self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

    def execute(self, requests: list["pb_utils.InferenceRequest"]) -> list["pb_utils.InferenceResponse"]:
        return [self.handle_request(request) for request in requests]

    def handle_request(self, request: "pb_utils.InferenceRequest") -> "pb_utils.InferenceResponse":
        tensor = pb_utils.get_input_tensor_by_name(request, "text")
        texts = [str_bytes.decode() for str_bytes in tensor.as_numpy()]
        encoded_texts = self._tokenize(texts)
        input_ids, attention_mask = encoded_texts["input_ids"], encoded_texts["attention_mask"]
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("input_ids", input_ids),
                pb_utils.Tensor("attention_mask", attention_mask),
            ]
        )
