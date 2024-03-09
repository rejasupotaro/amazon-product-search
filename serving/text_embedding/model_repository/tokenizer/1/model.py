import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def _tokenize(self, texts):
        return self._tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

    def process_request(self, request):
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

    def execute(self, requests):
        return [self.process_request(request) for request in requests]
