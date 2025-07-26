from unittest.mock import MagicMock, Mock, patch

import pytest

from amazon_product_search.retrieval.core.types import FieldType, ProcessedQuery, RetrievalConfig, SearchField
from amazon_product_search.retrieval.processors.base import (
    BaseQueryProcessor,
    ProcessorChain,
    SynonymExpandingProcessor,
)
from amazon_product_search.retrieval.processors.semantic import SemanticQueryProcessor
from amazon_product_search.retrieval.resources.manager import SharedResourceManager
from amazon_product_search.synonyms.synonym_dict import SynonymDict


@pytest.fixture
def retrieval_config():
    """Sample retrieval configuration."""
    return RetrievalConfig(
        index_name="products_jp",
        fields=[
            SearchField(name="title", field_type=FieldType.LEXICAL),
            SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
        ],
        size=20,
        enable_synonyms=True
    )


@pytest.fixture
def mock_synonym_dict():
    """Mock synonym dictionary."""
    synonym_dict = Mock(spec=SynonymDict)
    # Mock the look_up method that returns token chain format
    synonym_dict.look_up.return_value = [
        ("wireless", [("wireless", 1.0), ("bluetooth", 0.8), ("cordless", 0.7)]),
        ("headphones", [("headphones", 1.0), ("earphones", 0.7)])
    ]
    return synonym_dict


@pytest.fixture
def mock_resource_manager():
    """Mock resource manager."""
    manager = Mock(spec=SharedResourceManager)
    mock_encoder = Mock()
    mock_encoder.encode.return_value = Mock()
    mock_encoder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
    manager.get_encoder.return_value = mock_encoder
    return manager


class TestBaseQueryProcessor:
    """Test cases for BaseQueryProcessor."""

    def test_processor_initialization(self):
        """Test base processor initialization."""
        processor = BaseQueryProcessor(locale="jp")

        assert processor.locale == "jp"
        assert processor.tokenizer is not None
        assert processor.normalizer is not None

    def test_process_basic_query(self, retrieval_config):
        """Test processing a basic query."""
        processor = BaseQueryProcessor(locale="jp")
        query = processor.process("wireless headphones", retrieval_config)

        assert isinstance(query, ProcessedQuery)
        assert query.raw == "wireless headphones"
        assert query.normalized is not None
        assert len(query.tokens) > 0
        assert query.metadata is not None

    def test_process_empty_query(self, retrieval_config):
        """Test processing an empty query."""
        processor = BaseQueryProcessor(locale="jp")
        query = processor.process("", retrieval_config)

        assert query.raw == ""
        assert query.normalized == ""
        assert len(query.tokens) == 0

    def test_process_whitespace_query(self, retrieval_config):
        """Test processing a whitespace-only query."""
        processor = BaseQueryProcessor(locale="jp")
        query = processor.process("   \t\n   ", retrieval_config)

        assert query.raw == "   \t\n   "
        assert query.normalized == ""
        assert len(query.tokens) == 0

    def test_tokenization_japanese(self, retrieval_config):
        """Test tokenization for Japanese locale."""
        processor = BaseQueryProcessor(locale="jp")
        query = processor.process("無線ヘッドフォン", retrieval_config)

        assert query.raw == "無線ヘッドフォン"
        assert len(query.tokens) > 0
        assert isinstance(query.tokens, list)

    def test_tokenization_english(self, retrieval_config):
        """Test tokenization for English locale."""
        processor = BaseQueryProcessor(locale="us")
        query = processor.process("wireless headphones", retrieval_config)

        assert query.raw == "wireless headphones"
        assert len(query.tokens) == 2
        assert "wireless" in query.tokens
        assert "headphones" in query.tokens

    def test_metadata_collection(self, retrieval_config):
        """Test that processor collects metadata."""
        processor = BaseQueryProcessor(locale="jp")
        query = processor.process("test query", retrieval_config)

        assert "locale" in query.metadata
        assert "tokenized" in query.metadata
        assert query.metadata["locale"] == "jp"


class TestSynonymExpandingProcessor:
    """Test cases for SynonymExpandingProcessor."""

    def test_processor_initialization(self, mock_synonym_dict):
        """Test synonym processor initialization."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)

        assert processor.base_processor == base_processor
        assert processor.synonym_dict == mock_synonym_dict

    def test_process_with_synonyms(self, mock_synonym_dict, retrieval_config):
        """Test processing query with synonym expansion."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)

        query = processor.process("wireless headphones", retrieval_config)

        # Verify synonyms were retrieved
        mock_synonym_dict.look_up.assert_called_once_with(["wireless", "headphones"])

        # Verify synonyms were added to query
        assert isinstance(query, ProcessedQuery)
        assert query.synonyms is not None
        assert len(query.synonyms) > 0
        assert "bluetooth headphones" in query.synonyms
        assert "cordless headphones" in query.synonyms

    def test_process_synonyms_disabled(self, mock_synonym_dict, retrieval_config):
        """Test processing when synonyms are disabled in config."""
        retrieval_config.enable_synonyms = False

        base_processor = BaseQueryProcessor(locale="jp")
        processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)

        query = processor.process("wireless headphones", retrieval_config)

        # Should not call synonym dictionary
        mock_synonym_dict.look_up.assert_not_called()

        # Should not have synonyms
        assert query.synonyms is None or len(query.synonyms) == 0

    def test_process_no_synonyms_found(self, mock_synonym_dict, retrieval_config):
        """Test processing when no synonyms are found."""
        mock_synonym_dict.look_up.return_value = []

        base_processor = BaseQueryProcessor(locale="jp")
        processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)

        query = processor.process("wireless headphones", retrieval_config)

        # Should have empty synonyms list
        assert query.synonyms is not None
        assert len(query.synonyms) == 0

    def test_metadata_propagation(self, mock_synonym_dict, retrieval_config):
        """Test that metadata is properly propagated and augmented."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)

        query = processor.process("wireless headphones", retrieval_config)

        # Should have base metadata plus synonym metadata
        assert "locale" in query.metadata  # From base processor
        assert "synonym_expansions" in query.metadata
        assert query.metadata["synonym_expansions"] > 0


class TestSemanticQueryProcessor:
    """Test cases for SemanticQueryProcessor."""

    def test_processor_initialization(self, mock_resource_manager):
        """Test semantic processor initialization."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        assert processor.base_processor == base_processor
        assert processor.resource_manager == mock_resource_manager
        assert processor.model_name is not None

    def test_processor_initialization_without_resource_manager(self):
        """Test semantic processor initialization without resource manager."""
        base_processor = BaseQueryProcessor(locale="jp")

        with patch("amazon_product_search.retrieval.processors.semantic.SBERTEncoder") as mock_encoder:
            processor = SemanticQueryProcessor(base_processor=base_processor)

            assert processor.base_processor == base_processor
            assert processor.resource_manager is None
            assert processor.encoder is not None
            mock_encoder.assert_called_once()

    def test_process_with_vector_encoding(self, mock_resource_manager, retrieval_config):
        """Test processing query with vector encoding."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        query = processor.process("wireless headphones", retrieval_config)

        # Verify encoder was called
        mock_resource_manager.get_encoder.assert_called_once()
        mock_encoder = mock_resource_manager.get_encoder.return_value
        mock_encoder.encode.assert_called_once()

        # Verify vector was added to query
        assert query.vector is not None
        assert len(query.vector) == 4  # From mock encoder
        assert query.vector == [0.1, 0.2, 0.3, 0.4]

    def test_process_empty_normalized_query(self, mock_resource_manager, retrieval_config):
        """Test processing when normalized query is empty."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        query = processor.process("", retrieval_config)

        # Should not call encoder for empty query
        mock_resource_manager.get_encoder.assert_not_called()

        # Vector should be None or empty
        assert query.vector is None

    @patch("amazon_product_search.retrieval.processors.semantic.QueryVectorCache")
    def test_vector_caching(self, mock_cache_class, mock_resource_manager, retrieval_config):
        """Test that vector encoding uses caching."""
        # Mock cache instance
        mock_cache = MagicMock()
        mock_cache.__getitem__.return_value = None  # Cache miss
        mock_cache_class.return_value = mock_cache

        base_processor = BaseQueryProcessor(locale="jp")
        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        processor.process("wireless headphones", retrieval_config)

        # Verify cache was checked
        mock_cache.__getitem__.assert_called_once_with("wireless headphones")

    def test_metadata_vector_dimension(self, mock_resource_manager, retrieval_config):
        """Test that vector dimension is added to metadata."""
        base_processor = BaseQueryProcessor(locale="jp")
        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        query = processor.process("wireless headphones", retrieval_config)

        assert "vector_dim" in query.metadata
        assert query.metadata["vector_dim"] == 4  # From mock vector

    def test_custom_model_name(self, mock_resource_manager, retrieval_config):
        """Test processor with custom model name."""
        base_processor = BaseQueryProcessor(locale="jp")
        custom_model = "custom/model"

        processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager,
            model_name=custom_model
        )

        processor.process("test", retrieval_config)

        # Verify custom model was requested
        mock_resource_manager.get_encoder.assert_called_once_with(custom_model)


class TestProcessorChain:
    """Test cases for ProcessorChain."""

    def test_chain_initialization(self):
        """Test processor chain initialization."""
        processor1 = BaseQueryProcessor(locale="jp")
        processor2 = Mock()

        chain = ProcessorChain([processor1, processor2])

        assert len(chain.processors) == 2
        assert chain.processors[0] == processor1
        assert chain.processors[1] == processor2

    def test_chain_processing_order(self, retrieval_config):
        """Test that processors are called in order."""
        # Create mock processors that modify query
        processor1 = Mock()
        processor1.process.return_value = ProcessedQuery(
            raw="test", normalized="test", tokens=["test"], metadata={"step": 1}
        )

        processor2 = Mock()
        processor2.process.return_value = ProcessedQuery(
            raw="test", normalized="test", tokens=["test"], metadata={"step": 2}
        )

        chain = ProcessorChain([processor1, processor2])
        result = chain.process("test query", retrieval_config)

        # Verify both processors were called in order
        processor1.process.assert_called_once_with("test query", retrieval_config)
        processor2.process.assert_called_once()

        # Final result should be from processor2
        assert result.metadata["step"] == 2

    def test_empty_chain(self, retrieval_config):
        """Test empty processor chain."""
        chain = ProcessorChain([])

        # Should raise exception or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            chain.process("test query", retrieval_config)

    def test_single_processor_chain(self, retrieval_config):
        """Test chain with single processor."""
        processor = BaseQueryProcessor(locale="jp")
        chain = ProcessorChain([processor])

        result = chain.process("test query", retrieval_config)

        assert isinstance(result, ProcessedQuery)
        assert result.raw == "test query"


class TestProcessorIntegration:
    """Integration tests for query processors."""

    def test_full_processing_pipeline(self, mock_synonym_dict, mock_resource_manager, retrieval_config):
        """Test complete processing pipeline with all processors."""
        # Set up processors
        base_processor = BaseQueryProcessor(locale="jp")
        synonym_processor = SynonymExpandingProcessor(base_processor, mock_synonym_dict)
        semantic_processor = SemanticQueryProcessor(
            base_processor=synonym_processor,
            resource_manager=mock_resource_manager
        )

        chain = ProcessorChain([base_processor, synonym_processor, semantic_processor])

        # Process query
        result = chain.process("wireless headphones", retrieval_config)

        # Verify all processing steps were applied
        assert result.raw == "wireless headphones"
        assert result.normalized is not None
        assert len(result.tokens) > 0
        assert result.synonyms is not None
        assert result.vector is not None
        assert len(result.metadata) > 0

    def test_processor_error_handling(self, retrieval_config):
        """Test error handling in processor chain."""
        # Mock processor that raises exception
        failing_processor = Mock()
        failing_processor.process.side_effect = Exception("Processing error")

        chain = ProcessorChain([failing_processor])

        # Should propagate the exception
        with pytest.raises(Exception, match="Processing error"):
            chain.process("test query", retrieval_config)

    def test_processor_locale_handling(self):
        """Test that processors handle different locales correctly."""
        locales = ["jp", "us", "es"]

        for locale in locales:
            processor = BaseQueryProcessor(locale=locale)
            config = RetrievalConfig(
                index_name=f"products_{locale}",
                fields=[SearchField(name="title", field_type=FieldType.LEXICAL)],
                size=20
            )

            result = processor.process("test query", config)

            assert result.metadata["locale"] == locale

    def test_memory_efficiency(self, mock_resource_manager, retrieval_config):
        """Test that processors don't leak memory."""
        base_processor = BaseQueryProcessor(locale="jp")
        semantic_processor = SemanticQueryProcessor(
            base_processor=base_processor,
            resource_manager=mock_resource_manager
        )

        # Process multiple queries
        for i in range(10):
            query = semantic_processor.process(f"test query {i}", retrieval_config)
            assert query.vector is not None

        # Verify resource manager was reused
        assert mock_resource_manager.get_encoder.call_count >= 1
