import torch
from torch import Tensor, nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class Splade(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(bert_model_name)
        self.token_ids_to_skip = []
        for key in ["cls_token", "sep_token", "pad_token"]:
            token_to_skip = self.tokenizer.special_tokens_map[key]
            token_id_to_skip = self.tokenizer.vocab[token_to_skip]
            self.token_ids_to_skip.append(token_id_to_skip)

    @staticmethod
    def generate_bow(input_ids: Tensor, output_dim: int) -> Tensor:
        """from a batch of input ids, generates batch of bow rep"""
        bs = input_ids.shape[0]
        bow = torch.zeros(bs, output_dim)
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = 1
        return bow

    def encode_query(self, tokens: dict[str, Tensor]) -> Tensor:
        bow = self.generate_bow(tokens["input_ids"], self.tokenizer.vocab_size)
        for token_id_to_skip in self.token_ids_to_skip:
            bow[:, token_id_to_skip] = 0
        return bow

    def encode_doc(self, tokens: dict[str, Tensor]) -> Tensor:
        logits = self.model(**tokens).logits
        logits = torch.log(1 + torch.relu(logits))
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        bow = torch.sum(logits * attention_mask, dim=1)
        return bow

    def forward(self, query: dict[str, Tensor], doc: dict[str, Tensor]) -> Tensor:
        encoded_doc = self.encode_doc(doc)
        encoded_query = self.encode_query(query).to(encoded_doc.device)
        score = torch.sum(encoded_query * encoded_doc, dim=1, keepdim=True)
        return score
