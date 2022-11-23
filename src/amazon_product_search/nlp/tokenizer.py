from enum import Enum, auto
from typing import Union

import ipadic
from fugashi import GenericTagger, Tagger

TAGGER = Union[Tagger, GenericTagger]


class TokenizerType(Enum):
    UNIDIC = auto()
    IPADIC = auto()


class Tokenizer:
    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.UNIDIC):
        self.tagger: Tagger
        if tokenizer_type == TokenizerType.UNIDIC:
            self.tagger = Tagger("-Owakati")
        elif tokenizer_type == TokenizerType.IPADIC:
            self.tagger = GenericTagger(f"-Owakati {ipadic.MECAB_ARGS}")
        else:
            raise ValueError(f"Unsupported tokenizer_type was given: {tokenizer_type}")

    def tokenize(self, s: str) -> list[str]:
        """Tokenize a given string into tokens.

        Args:
            s (str): A string to tokenize.

        Returns:
            List[str]: A resulting of tokens.
        """
        return self.tagger.parse(s).split()
