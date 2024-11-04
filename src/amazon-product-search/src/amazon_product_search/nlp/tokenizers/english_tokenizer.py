from amazon_product_search.nlp.tokenizers.tokenizer import Tokenizer


class EnglishTokenizer(Tokenizer):
    def tokenize(self, s: str) -> list[str] | list[tuple[str, str]]:
        return s.split()
