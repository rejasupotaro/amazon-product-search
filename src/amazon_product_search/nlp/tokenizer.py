from enum import Enum, auto
from typing import Union

import ipadic
from fugashi import GenericTagger, Tagger

TAGGER = Union[Tagger, GenericTagger]


class TokenizerType(Enum):
    UNIDIC = auto()
    IPADIC = auto()


# https://hayashibe.jp/tr/mecab/dictionary/unidic/pos (UniDic)
# What is "形状詞" in English?
class POSTag(Enum):
    NOUN = "名詞"
    PRONOUN = "代名詞"
    ADNOMINAL = "連体詞"
    ADVERB = "副詞"
    CONJUNCTION = "接続詞"
    INTERJECTION = "感動詞"
    VERB = "動詞"
    ADJECTIVE = "形容詞"
    AUXILIARY_VERB = "助動詞"
    PARTICLE = "助詞"
    PREFIX = "接頭辞"
    SUFFIX = "接尾辞"
    SYNBOL = "記号"
    AUXILIARY_SYMBOL = "補助記号"
    WHITESPACE = "空白"


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
