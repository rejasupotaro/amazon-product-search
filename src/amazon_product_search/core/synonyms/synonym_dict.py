import re
from collections import defaultdict
from typing import cast

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
        threshold: float = 0.8,
        skip_numbers: bool = True,
    ) -> None:
        self.tokenizer = locale_to_tokenizer(locale)
        self.threshold = threshold
        if synonym_filename:
            self._entry_dict: dict[str, list[tuple[str, float]]] = self.load_synonym_dict(
                data_dir,
                synonym_filename,
                skip_numbers,
            )
        else:
            self._entry_dict = defaultdict(list)

    def load_synonym_dict(
        self, data_dir: str, synonym_filename: str, skip_numbers: bool
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
            query: str = row["query"]
            title: str = row["title"]
            if skip_numbers and (has_numbers(query) or has_numbers(title)):
                continue

            similarity = row["similarity"]
            if similarity > self.threshold:
                continue

            entry_dict[query].append((title, similarity))
        return entry_dict

    def find_synonyms(self, query: str) -> list[str]:
        """Return a list of synonyms for a given query.

        Args:
            query (str): A query to expand.

        Returns:
            list[str]: A list of synonyms.
        """
        all_synonyms = []
        tokens = cast(list, self.tokenizer.tokenize(query))
        for token in tokens:
            if token not in self._entry_dict:
                continue
            candidates: list[tuple[str, float]] = self._entry_dict[token]
            synonyms = [synonym for synonym, _ in candidates]
            if synonyms:
                all_synonyms.extend(synonyms)
        return all_synonyms
