from collections import defaultdict

import polars as pl

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.core.nlp.tokenizer import Tokenizer


class SynonymDict:
    def __init__(
        self, data_dir: str = DATA_DIR, synonym_filename: str = "synonyms_jp_sbert.csv"
    ) -> None:
        self.tokenizer = Tokenizer()
        self._entry_dict: dict[str, list[tuple[str, float]]] = self.load_synonym_dict(
            data_dir, synonym_filename
        )

    @staticmethod
    def load_synonym_dict(
        data_dir: str, synonym_filename: str
    ) -> dict[str, list[tuple[str, float]]]:
        """Load the synonym file and convert it into a dict for lookup.

        Args:
            data_dir (str): The data directory.
            synonym_filename (str): A filename to load, which is supposed to be under `{DATA_DIR}/includes`.

        Returns:
            dict[str, list[str]]: The converted synonym dict.
        """
        df = pl.read_csv(f"{data_dir}/includes/{synonym_filename}")
        entry_dict = defaultdict(list)
        for row in df.to_dicts():
            query = row["query"]
            title = row["title"]
            similarity = row["similarity"]
            entry_dict[query].append((title, similarity))
        # df = df[df["similarity"] >= threshold]
        # queries = df["query"].tolist()
        # synonyms = df["title"].tolist()
        # entry_dict = defaultdict(list)
        # for query, synonym in zip(queries, synonyms):
        #     entry_dict[query].append(synonym)
        return entry_dict

    def find_synonyms(self, query: str, threshold: float = 0.6) -> list[str]:
        """Return a list of synonyms for a given query.

        Args:
            query (str): A query to expand.
            threshold (float, optional): A threshold for the confidence (similarity) of synonyms. Defaults to 0.6.

        Returns:
            list[str]: A list of synonyms.
        """
        all_synonyms = []
        tokens = self.tokenizer.tokenize(query)
        for token in tokens:
            if token not in self._entry_dict:
                continue
            candidates: list[tuple[str, float]] = self._entry_dict[token]
            synonyms = [
                synonym for synonym, similarity in candidates if similarity >= threshold
            ]
            if synonyms:
                all_synonyms.extend(synonyms)
        return all_synonyms
