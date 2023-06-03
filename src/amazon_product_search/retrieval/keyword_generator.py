import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class KeywordGenerator:
    def __init__(
        self, model_name: str = "doc2query/msmarco-japanese-mt5-base-v1"
    ) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, text: str, num_queries: int = 10) -> list[str]:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=256,
            do_sample=True,
            top_k=10,
            num_return_sequences=num_queries,
        )
        return list(
            {
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                for i in range(num_queries)
            }
        )
