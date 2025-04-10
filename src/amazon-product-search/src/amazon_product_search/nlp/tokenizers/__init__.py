from data_source import Locale

from amazon_product_search.nlp.tokenizers.english_tokenizer import EnglishTokenizer
from amazon_product_search.nlp.tokenizers.japanese_tokenizer import JapaneseTokenizer
from amazon_product_search.nlp.tokenizers.tokenizer import Tokenizer


def locale_to_tokenizer(locale: Locale) -> Tokenizer:
    return {
        "us": EnglishTokenizer,
        "jp": JapaneseTokenizer,
    }[locale]()


__all__ = [
    "EnglishTokenizer",
    "JapaneseTokenizer",
    "Tokenizer",
]
