import re
from collections import defaultdict

import polars as pl

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.core.nlp.tokenizers import locale_to_tokenizer
from amazon_product_search.core.source import Locale


def has_numbers(s: str) -> bool:
    return bool(re.search(r"\d", s))


class SynonymDict:
    def __init__(
        self,
        locale: Locale,
        synonym_filename: str | None = None,
        data_dir: str = DATA_DIR,
        threshold: float = 0.6,
    ) -> None:
        self.tokenizer = locale_to_tokenizer(locale)
        self.threshold = threshold
        if synonym_filename:
            self._entry_dict: dict[str, list[tuple[str, float]]] = self.load_synonym_dict(
                data_dir,
                synonym_filename,
            )
        else:
            self._entry_dict = defaultdict(list)

    def load_synonym_dict(self, data_dir: str, synonym_filename: str) -> dict[str, list[tuple[str, float]]]:
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
            query: str = row["query"]
            title: str = row["title"]
            score: float = row["npmi"] * row["similarity"]
            if score > self.threshold:
                continue

            entry_dict[query].append((title, score))
        return entry_dict

    def expand_synonyms(self, tokens: list[str], ngrams: int = 2) -> list[tuple[str, list[str]]]:
        """Return a list of synonyms for a given query.

        Args:
            query (str): A query to expand.

        Returns:
            list[tuple[str, list[str]]]: A list of the original query terms and their synonyms.
        """
        expanded_query: list[tuple[str, list[str]]] = []
        current_position = 0
        while current_position < len(tokens):
            found = None
            for n in range(ngrams, 0, -1):
                end_position = current_position + n
                if end_position > len(tokens):
                    continue
                key = " ".join(tokens[current_position:end_position])
                if key not in self._entry_dict:
                    continue
                candidates: list[tuple[str, float]] = self._entry_dict[key]
                synonym = [candidate[0] for candidate in candidates]
                found = (key, synonym)
                break
            if found:
                expanded_query.append(found)
                current_position = end_position
            else:
                expanded_query.append((tokens[current_position], []))
                current_position += 1
        return expanded_query
