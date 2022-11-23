from enum import Enum, auto
from typing import TypeAlias

import ipadic
from fugashi import GenericTagger, Tagger

TAGGER: TypeAlias = Tagger | GenericTagger


class DicType(Enum):
    UNIDIC = auto()
    IPADIC = auto()


class OutputFormat(Enum):
    WAKATI = "wakati"
    DUMP = "dump"


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
    def __init__(self, dic_type: DicType = DicType.UNIDIC, output_format: OutputFormat = OutputFormat.WAKATI):
        self.dic_type = dic_type
        self.output_format = output_format

        tagger_options = []
        if dic_type == DicType.IPADIC:
            tagger_options.append(ipadic.MECAB_ARGS)
        if output_format == OutputFormat.WAKATI:
            tagger_options.append(f"-O{output_format.value}")

        self.tagger: Tagger
        match dic_type:
            case DicType.UNIDIC:
                self.tagger = Tagger(" ".join(tagger_options))
            case DicType.IPADIC:
                self.tagger = GenericTagger(" ".join(tagger_options))
            case _:
                raise ValueError(f"Unsupported dic_type was given: {dic_type}")

    def tokenize(self, s: str) -> list[str | tuple[str, str]]:
        """Tokenize a given string into tokens.

        Args:
            s (str): A string to tokenize.

        Returns:
            list[str | tuple[str, str]]: A resulting of tokens.
                If output_format is WAKATI, return a list of tokens.
                If output_format is DUMP, return a list of tuples of (token, POS tags).
        """
        if self.output_format == OutputFormat.WAKATI:
            return self.tagger.parse(s).split()

        tokens = []
        pos_tags = []
        for result in self.tagger(s):
            tokens.append(str(result))
            match self.dic_type:
                case DicType.UNIDIC:
                    pos_tags.append(result.pos.split(","))
                case DicType.IPADIC:
                    pos_tags.append(result.feature)
        return list(zip(tokens, pos_tags))
