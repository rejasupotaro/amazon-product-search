
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
)


class TokenizedSentencesDataset(Dataset):
    def __init__(self, sentences: list[str], tokenizer: AutoTokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.tokenizer(
            self.sentences[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )
