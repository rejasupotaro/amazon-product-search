from typing import List

import ipadic
from fugashi import GenericTagger


class Tokenizer:
    def __init__(self):
        self.tagger = GenericTagger(f"-Owakati {ipadic.MECAB_ARGS}")

    def tokenize(self, s: str) -> List[str]:
        """Tokenize a given string into tokens.

        Args:
            s (str): A string to tokenize.

        Returns:
            List[str]: A resulting of tokens.
        """
        return self.tagger.parse(s).split()
