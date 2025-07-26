import logging
from typing import Sequence, cast

from data_source import Locale

from amazon_product_search.nlp.normalizer import normalize_query
from amazon_product_search.nlp.tokenizers import Tokenizer, locale_to_tokenizer
from amazon_product_search.retrieval.core.protocols import QueryProcessor
from amazon_product_search.retrieval.core.types import ProcessedQuery, RetrievalConfig
from amazon_product_search.synonyms.synonym_dict import SynonymDict, expand_synonyms

logger = logging.getLogger(__name__)


class BaseQueryProcessor(QueryProcessor):
    """Base implementation of query processing with normalization and tokenization."""

    def __init__(self, locale: Locale):
        self.locale = locale
        self.tokenizer: Tokenizer = locale_to_tokenizer(locale)
        self.normalizer = normalize_query  # Expose normalizer function

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Process a raw query into structured format."""
        normalized = normalize_query(raw_query)
        tokens = cast(list[str], self.tokenizer.tokenize(normalized))

        return ProcessedQuery(
            raw=raw_query,
            normalized=normalized,
            tokens=tokens,
            metadata={"locale": self.locale, "tokenized": True}
        )


class SynonymExpandingProcessor(QueryProcessor):
    """Query processor that expands synonyms."""

    def __init__(self, base_processor: QueryProcessor, synonym_dict: SynonymDict):
        self.base_processor = base_processor
        self.synonym_dict = synonym_dict

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Process query and expand with synonyms if enabled."""
        query = self.base_processor.process(raw_query, config)

        if not config.enable_synonyms or not self.synonym_dict:
            return query

        # Expand synonyms
        token_chain = self.synonym_dict.look_up(query.tokens)

        if not token_chain:
            # No synonyms found
            synonyms = []
        else:
            expanded_tokens: list[list[tuple[str, float | None]]] = []
            expand_synonyms(
                [((token, None), synonym_score_tuples) for token, synonym_score_tuples in token_chain],
                [],
                expanded_tokens,
            )

            # Take first 10 expansions and convert to string list
            synonyms = [" ".join([token for token, _ in token_scores])
                       for token_scores in expanded_tokens[:10]]
            # Remove empty strings
            synonyms = [s for s in synonyms if s.strip()]

        query.synonyms = synonyms
        query.metadata["synonym_expansions"] = len(synonyms)

        return query


class ProcessorChain(QueryProcessor):
    """Chain multiple query processors together."""

    def __init__(self, processors: Sequence[QueryProcessor]):
        self.processors = list(processors)

    def process(self, raw_query: str, config: RetrievalConfig) -> ProcessedQuery:
        """Apply all processors in sequence."""
        if not self.processors:
            raise ValueError("ProcessorChain requires at least one processor")

        query = self.processors[0].process(raw_query, config)

        for processor in self.processors[1:]:
            # For chained processors, we pass the processed query as the "raw" query
            # This allows processors to build on each other's work
            query = processor.process(query.normalized, config)
            # Preserve original query reference
            query.raw = raw_query

        return query
