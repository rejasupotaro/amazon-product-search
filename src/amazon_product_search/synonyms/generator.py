from collections import Counter
from math import log

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from amazon_product_search.constants import DATA_DIR
from amazon_product_search.nlp.normalizer import normalize_doc
from amazon_product_search.nlp.tokenizer import Tokenizer
from amazon_product_search.source import load_merged
from amazon_product_search.synonyms.filters.similarity_filter import SimilarityFilter


def load_query_title_pairs(locale: str, nrows: int = -1) -> list[list[str]]:
    """Load query title pairs."""
    df = load_merged(locale, nrows)
    df = df[df["esci_label"] == "exact"]
    df = df[(~df["query"].isna()) & (~df["product_title"].isna())]
    pairs = df[["query", "product_title"]].values.tolist()
    return pairs


def generate_candidates(pairs: list[list[str]], min_cooccurrence: int = 10, min_npmi: float = 0.5) -> pd.DataFrame:
    """Generate synonyms based on cooccurrence."""
    tokenizer = Tokenizer()
    word_counter: Counter = Counter()
    pair_counter: Counter = Counter()

    print("Counting words...")
    for query, title in tqdm(pairs):
        query = normalize_doc(query)
        title = normalize_doc(title)
        if not query or not title:
            continue

        for query_word in tokenizer.tokenize(query):
            word_counter[query_word] += 1
            for title_word in tokenizer.tokenize(title):
                word_counter[title_word] += 1
                if query_word != title_word:
                    pair_counter[(query_word, title_word)] += 1

    total_word_count = sum(word_counter.values())
    log_total_word_count = log(total_word_count)

    print("Calculating metrics...")
    rows = []
    for key in tqdm(pair_counter):
        query, title = key
        query_count, title_count, pair_count = (
            word_counter[query],
            word_counter[title],
            pair_counter[key],
        )
        log_pair_count, log_query_count, log_title_count = (
            log(pair_count),
            log(query_count),
            log(title_count),
        )

        pmi = log_total_word_count + log_pair_count - log_query_count - log_title_count
        npmi = pmi / (log_total_word_count - log_pair_count)

        rows.append(
            {
                "query": query,
                "title": title,
                "query_count": query_count,
                "title_count": title_count,
                "cooccurrence": pair_count,
                "npmi": npmi,
            }
        )

    candidates_df = DataFrame(rows)
    candidates_df = candidates_df[candidates_df["cooccurrence"] >= min_cooccurrence]
    candidates_df = candidates_df[candidates_df["npmi"].abs() >= min_npmi]
    return candidates_df


def generate():
    """Generate synonyms from query title pairs.

    1. Load the relevance judgement file.
    2. Filter rows with ESCI label: "exact".
    3. Extract query title pairs.
    4. Calculate word cooccurrence and npmi and filter out those with low scores.
    5. The filtered candidates are further filtered by cosine similarity.
    6. Save the generated synonyms to `{DATA_DIR}/synonyms.csv`.
    """
    pairs = load_query_title_pairs(locale="jp", nrows=100)
    print(f"{len(pairs)} title pairs will be processed")

    candidates_df = generate_candidates(pairs)
    print(f"{len(candidates_df)} candidates were generated")

    filter = SimilarityFilter()
    synonyms_df = filter.apply(candidates_df)
    print(f"{len(synonyms_df)} synonyms were generated")

    filepath = f"{DATA_DIR}/synonyms.csv"
    synonyms_df.to_csv(filepath, index=False)
