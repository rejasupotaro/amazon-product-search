from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, s: str) -> list[str | tuple[str, str]]:
        ...
