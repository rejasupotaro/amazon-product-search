from enum import Enum, auto

import ipadic
from fugashi import GenericTagger, Tagger


class DicType(Enum):
    IPADIC = auto()
    UNIDIC = auto()


class Tokenizer:
    def __init__(self, dic_type: DicType = DicType.IPADIC):
        if dic_type == DicType.IPADIC:
            self.tagger = GenericTagger(f"-Owakati {ipadic.MECAB_ARGS}")
        elif dic_type == DicType.UNIDIC:
            self.tagger = Tagger(f"-Owakati")
        else:
            raise ValueError(f"Unsupported dic_type was given: {dic_type}")

    def tokenize(self, s: str) -> list[str]:
        """Tokenize a given string into tokens.

        Args:
            s (str): A string to tokenize.

        Returns:
            List[str]: A resulting of tokens.
        """
        return self.tagger.parse(s).split()
