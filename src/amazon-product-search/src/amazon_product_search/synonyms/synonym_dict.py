import re
from collections import defaultdict

import pandas as pd
from data_source import Locale

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.nlp.tokenizers import locale_to_tokenizer


def has_numbers(s: str) -> bool:
    return bool(re.search(r"\d", s))


def expand_synonyms(
    token_chain: list[tuple[tuple[str, float | None], list[tuple[str, float]]]],
    current: list[tuple[str, float | None]],
    result: list[list[tuple[str, float | None]]],
):
    """Expand synonyms recursively.

    This function expands synonyms recursively and appends the result to the result list.
    For example, if the input token_chain is [("a", None), [("b", None), [("b'", 1.0)]]],
    the output result will be:
    [
        [("a", None), ("b", None)],
        [("a", None), ("b'", 1.0)],
    ]

    Args:
        token_chain (list[tuple[str, list[tuple[str, float]]]]): A list of tokens and their synonyms.
        current (list[tuple[str, float]]): The current list of tokens and their synonyms.
        result (list[list[tuple[str, float]]]): The result list of expanded synonyms with scores.
    """
    if not token_chain:
        result.append(current)
        return

    token, synonym_score_tuples = token_chain[0]
    expand_synonyms(token_chain[1:], [*current, token], result)
    if synonym_score_tuples:
        for synonym, score in synonym_score_tuples:
            expand_synonyms(token_chain[1:], [*current, (synonym, score)], result)


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
        df = pd.read_csv(f"{data_dir}/includes/{synonym_filename}")
        entry_dict = defaultdict(list)
        for row in df.to_dict("records"):
            query: str = row["query"]
            title: str = row["title"]
            score: float = row["npmi"] * row["similarity"]
            if score > self.threshold:
                continue

            entry_dict[query].append((title, score))
        return entry_dict

    def look_up(self, tokens: list[str], ngrams: int = 2) -> list[tuple[str, list[tuple[str, float]]]]:
        """Return a list of synonyms for a given query.

        Args:
            tokens (list[str]): A list of tokens to look up.

        Returns:
            list[tuple[str, list[str]]]: A list of the original query terms and their synonyms.
        """
        expanded_query: list[tuple[str, list[tuple[str, float]]]] = []
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
                synonyms = self._entry_dict[key]
                found = (key, synonyms)
                break
            if found:
                expanded_query.append(found)
                current_position = end_position
            else:
                expanded_query.append((tokens[current_position], []))
                current_position += 1
        return expanded_query
