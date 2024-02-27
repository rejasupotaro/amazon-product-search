import itertools
from collections import Counter
from math import log

import polars as pl
from tqdm import tqdm

from amazon_product_search.constants import DATA_DIR, HF
from amazon_product_search.core.nlp.normalizer import normalize_doc
from amazon_product_search.core.nlp.tokenizers import locale_to_tokenizer
from amazon_product_search.core.source import Locale, load_merged
from amazon_product_search.core.synonyms.filters import utils
from amazon_product_search.core.synonyms.filters.similarity_filter import SimilarityFilter


def load_query_title_pairs(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    """Load query title pairs."""
    df = load_merged(locale, nrows)
    df = df.filter(pl.col("esci_label") == "E")
    return df


def preprocess_query_title_pairs(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter((pl.col("query").is_not_null() & pl.col("product_title").is_not_null()))
    return df.with_columns(
        [
            pl.col("query").apply(normalize_doc),
            pl.col("product_title").apply(normalize_doc),
        ]
    )


def generate_ngrams(tokens: list[str], n: int) -> list[str]:
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        ngrams.append(ngram)
    return ngrams


def generate_ngrams_all(tokens: list[str], n: int) -> list[str]:
    ngrams = []
    for i in range(1, n + 1):
        ngrams.extend(generate_ngrams(tokens, i))
    return ngrams


def count_words(
    locale: Locale,
    pairs: list[list[str]],
    ngrams: int = 2,
) -> tuple[Counter, Counter]:
    """Generate synonyms based on cooccurrence."""
    tokenizer = locale_to_tokenizer(locale)
    word_counter: Counter = Counter()
    pair_counter: Counter = Counter()

    print("Counting words...")
    for query, title in tqdm(pairs):
        if not query or not title:
            continue

        query_tokens = [
            token for token in tokenizer.tokenize(query) if isinstance(token, str) and token not in utils.STOPWORDS
        ]
        query_tokens = generate_ngrams_all(query_tokens, ngrams)
        title_tokens = [
            token for token in tokenizer.tokenize(title) if isinstance(token, str) and token not in utils.STOPWORDS
        ]
        title_tokens = generate_ngrams_all(title_tokens, ngrams)
        for query_word, title_word in itertools.product(query_tokens, title_tokens):
            word_counter[query_word] += 1
            word_counter[title_word] += 1
            if query_word != title_word:
                pair_counter[(query_word, title_word)] += 1
    return word_counter, pair_counter


def apply_fast_filters(
    word_counter: Counter,
    pair_counter: Counter,
    min_cooccurrence: int,
    min_npmi: float,
) -> pl.DataFrame:
    total_word_count = sum(word_counter.values())
    log_total_word_count = log(total_word_count)

    print("Calculating metrics...")
    rows = []
    original_num_candidates = len(pair_counter)
    for key in tqdm(pair_counter):
        query, title = key

        if utils.are_two_sets_identical(query, title):
            continue
        if utils.is_either_contained_in_other(query, title):
            continue
        if utils.is_either_number(query, title):
            continue

        query_count, title_count, pair_count = (
            word_counter[query],
            word_counter[title],
            pair_counter[key],
        )
        if pair_count < min_cooccurrence:
            continue

        log_pair_count, log_query_count, log_title_count = (
            log(pair_count),
            log(query_count),
            log(title_count),
        )

        pmi = log_total_word_count + log_pair_count - log_query_count - log_title_count
        npmi = pmi / (log_total_word_count - log_pair_count)
        if npmi < min_npmi:
            continue

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
    filtered_num_candidates = len(rows)
    diff = original_num_candidates - filtered_num_candidates
    print(f"Filtered out {diff} candidates ({original_num_candidates} -> {filtered_num_candidates})")

    candidates_df = pl.from_dicts(rows)
    return candidates_df


def generate(
    locale: Locale,
    output_filename: str,
    min_cooccurrence: int = 10,
    min_npmi: float = 0.5,
) -> None:
    """Generate synonyms from query title pairs.

    1. Load the relevance judgement file.
    2. Filter rows with ESCI label: "exact".
    3. Extract query title pairs.
    4. Calculate word cooccurrence and npmi and filter out those with low scores.
    5. The filtered candidates are further filtered by cosine similarity.
    6. Save the generated synonyms to `{DATA_DIR}/includes/{output_filename}`.

    Args:
        locale (Locale): The target locale.
        output_filename (str): The output filename.
    """
    hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]

    print("Load query-title pairs")
    pairs_df = load_query_title_pairs(locale=locale)
    print(f"{len(pairs_df)} pairs will be processed")

    print("Preprocess query-title pairs")
    pairs_df = preprocess_query_title_pairs(pairs_df)

    print("Generate candidates from query-title pairs")
    pairs = pairs_df.select(["query", "product_title"]).to_numpy().tolist()
    word_counter, pair_counter = count_words(locale, pairs)
    candidates_df = apply_fast_filters(word_counter, pair_counter, min_cooccurrence, min_npmi)
    print(f"{len(candidates_df)} candidates were generated")

    print("Filter synonyms by Semantic Similarity")
    filter = SimilarityFilter(hf_model_name)
    synonyms_df = filter.apply(candidates_df)

    filepath = f"{DATA_DIR}/includes/{output_filename}"
    synonyms_df.write_csv(filepath)
    print(f"{len(synonyms_df)} synonyms were saved to {filepath}")
