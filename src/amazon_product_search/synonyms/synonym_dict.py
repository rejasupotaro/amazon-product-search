from collections import defaultdict

import pandas as pd

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.nlp.tokenizer import Tokenizer


class SynonymDict:
    def __init__(self, synonym_filename: str = "synonyms_jp_sbert.csv", threshold: float = 0.6):
        self.tokenizer = Tokenizer()
        self._entry_dict = self.load_synonym_dict(synonym_filename, threshold)

    @staticmethod
    def load_synonym_dict(synonym_filename: str, threshold: float) -> dict[str, list[str]]:
        """Load the synonym file and convert it into a dict for lookup.

        Args:
            threshold (float, optional): A threshold for the confidence (similarity) of synonyms. Defaults to 0.6.

        Returns:
            dict[str, list[str]]: The converted synonym dict.
        """
        df = pd.read_csv(f"{DATA_DIR}/includes/{synonym_filename}")
        df = df[df["similarity"] >= threshold]
        queries = df["query"].tolist()
        synonyms = df["title"].tolist()
        entry_dict = defaultdict(list)
        for query, synonym in zip(queries, synonyms):
            entry_dict[query].append(synonym)
        return entry_dict

    def find_synonyms(self, query: str) -> list[str]:
        """Return a list of synonyms for a given query.

        Args:
            query (str): A query to expand.

        Returns:
            list[str]: A list of synonyms.
        """
        synonyms = []
        tokens = self.tokenizer.tokenize(query)
        for token in tokens:
            if token not in self._entry_dict:
                continue
            synonyms.extend(self._entry_dict[token])
        return synonyms
