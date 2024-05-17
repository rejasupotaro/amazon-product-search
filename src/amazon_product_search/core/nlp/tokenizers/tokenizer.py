from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, s: str) -> list[str] | list[tuple[str, str]]: ...
