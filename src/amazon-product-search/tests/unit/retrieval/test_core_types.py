from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.types import (
    FieldType,
    FusionConfig,
    ProcessedQuery,
    RetrievalConfig,
    RetrievalResponse,
    SearchField,
)
from amazon_product_search.retrieval.response import Response, Result
from amazon_product_search.retrieval.retriever import Retriever
from amazon_product_search.synonyms.synonym_dict import SynonymDict


class TestFieldType:
    """Test cases for FieldType enum."""

    def test_field_type_values(self):
        """Test that FieldType enum has correct values."""
        assert FieldType.LEXICAL.value == "lexical"
        assert FieldType.SEMANTIC.value == "semantic"

    def test_field_type_comparison(self):
        """Test FieldType comparison."""
        assert FieldType.LEXICAL == FieldType.LEXICAL
        assert FieldType.SEMANTIC == FieldType.SEMANTIC
        assert FieldType.LEXICAL != FieldType.SEMANTIC


class TestSearchField:
    """Test cases for SearchField dataclass."""

    def test_search_field_creation(self):
        """Test SearchField creation with all parameters."""
        field = SearchField(
            name="product_title",
            field_type=FieldType.LEXICAL,
            weight=2.0
        )

        assert field.name == "product_title"
        assert field.field_type == FieldType.LEXICAL
        assert field.weight == 2.0

    def test_search_field_defaults(self):
        """Test SearchField with default values."""
        field = SearchField(name="title", field_type=FieldType.LEXICAL)

        assert field.name == "title"
        assert field.field_type == FieldType.LEXICAL
        assert field.weight == 1.0  # Default weight

    def test_search_field_immutable(self):
        """Test that SearchField is immutable (frozen dataclass)."""
        field = SearchField(name="title", field_type=FieldType.LEXICAL)

        with pytest.raises(FrozenInstanceError):
            field.name = "new_title"

        with pytest.raises(FrozenInstanceError):
            field.weight = 2.0


class TestProcessedQuery:
    """Test cases for ProcessedQuery dataclass."""

    def test_processed_query_creation(self):
        """Test ProcessedQuery creation with all fields."""
        query = ProcessedQuery(
            raw="wireless headphones",
            normalized="wireless headphones",
            tokens=["wireless", "headphones"],
            vector=[0.1, 0.2, 0.3],
            synonyms=["bluetooth headphones"],
            metadata={"locale": "jp"}
        )

        assert query.raw == "wireless headphones"
        assert query.normalized == "wireless headphones"
        assert query.tokens == ["wireless", "headphones"]
        assert query.vector == [0.1, 0.2, 0.3]
        assert query.synonyms == ["bluetooth headphones"]
        assert query.metadata == {"locale": "jp"}

    def test_processed_query_defaults(self):
        """Test ProcessedQuery with default values."""
        query = ProcessedQuery(
            raw="test",
            normalized="test",
            tokens=["test"]
        )

        assert query.raw == "test"
        assert query.normalized == "test"
        assert query.tokens == ["test"]
        assert query.vector is None
        assert query.synonyms is None
        assert query.metadata == {}

    def test_processed_query_empty_metadata(self):
        """Test ProcessedQuery with empty metadata."""
        query = ProcessedQuery(
            original="test",
            normalized="test",
            tokens=["test"],
            metadata={}
        )

        assert isinstance(query.metadata, dict)
        assert len(query.metadata) == 0


class TestRetrievalConfig:
    """Test cases for RetrievalConfig dataclass."""

    def test_retrieval_config_creation(self):
        """Test RetrievalConfig creation with all parameters."""
        fields = [
            SearchField(name="title", field_type=FieldType.LEXICAL),
            SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
        ]

        config = RetrievalConfig(
            index_name="products_jp",
            fields=fields,
            size=20,
            window_size=50,
            enable_synonyms=True,
            enable_explain=True,
            filters={"category": "electronics"}
        )

        assert config.index_name == "products_jp"
        assert len(config.fields) == 2
        assert config.size == 20
        assert config.window_size == 50
        assert config.enable_synonyms is True
        assert config.enable_explain is True
        assert config.filters == {"category": "electronics"}

    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig with default values."""
        fields = [SearchField(name="title", field_type=FieldType.LEXICAL)]

        config = RetrievalConfig(
            index_name="test_index",
            fields=fields,
            size=10
        )

        assert config.index_name == "test_index"
        assert config.size == 10
        assert config.window_size is None
        assert config.enable_synonyms is False
        assert config.enable_explain is False
        assert config.filters == {}

    def test_get_fields_by_type(self):
        """Test get_fields_by_type method."""
        fields = [
            SearchField(name="title", field_type=FieldType.LEXICAL),
            SearchField(name="description", field_type=FieldType.LEXICAL),
            SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
        ]

        config = RetrievalConfig(
            index_name="test_index",
            fields=fields,
            size=10
        )

        lexical_fields = config.get_fields_by_type(FieldType.LEXICAL)
        semantic_fields = config.get_fields_by_type(FieldType.SEMANTIC)

        assert len(lexical_fields) == 2
        assert len(semantic_fields) == 1
        assert lexical_fields[0].name == "title"
        assert lexical_fields[1].name == "description"
        assert semantic_fields[0].name == "title_vector"


class TestRetrievalResponse:
    """Test cases for RetrievalResponse dataclass."""

    def test_retrieval_response_creation(self):
        """Test RetrievalResponse creation."""
        results = [
            Result(product={"product_id": "123", "title": "Test"}, score=0.9)
        ]

        response = RetrievalResponse(
            results=results,
            total_hits=1,
            engine_name="test_engine",
            processing_time_ms=15.0,
            metadata={"test": True}
        )

        assert len(response.results) == 1
        assert response.total_hits == 1
        assert response.engine_name == "test_engine"
        assert response.processing_time_ms == 15.0
        assert response.metadata == {"test": True}

    def test_retrieval_response_defaults(self):
        """Test RetrievalResponse with default values."""
        response = RetrievalResponse(
            results=[],
            total_hits=0,
            engine_name="test"
        )

        assert response.results == []
        assert response.total_hits == 0
        assert response.engine_name == "test"
        assert response.processing_time_ms == 0.0
        assert response.metadata == {}


class TestFusionConfig:
    """Test cases for FusionConfig dataclass."""

    def test_fusion_config_creation(self):
        """Test FusionConfig creation."""
        config = FusionConfig(
            method="rrf",
            normalization="min_max",
            weights={"lexical": 0.7, "semantic": 1.3}
        )

        assert config.method == "rrf"
        assert config.normalization == "min_max"
        assert config.weights == {"lexical": 0.7, "semantic": 1.3}

    def test_fusion_config_defaults(self):
        """Test FusionConfig with default values."""
        config = FusionConfig(method="weighted_sum")

        assert config.method == "weighted_sum"
        assert config.normalization == "none"
        assert config.weights == {}


class TestModernizedRetriever:
    """Test cases for the modernized Retriever class."""

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_retriever_initialization(self, mock_es_client, mock_resource_manager):
        """Test Retriever initialization with new architecture."""
        retriever = Retriever(locale="jp")

        assert retriever.locale == "jp"
        assert retriever.es_client is not None
        assert retriever.resource_manager is not None
        assert retriever.query_processor is not None
        assert len(retriever.retrieval_engines) == 2  # Lexical and semantic
        assert retriever.result_fuser is not None
        assert retriever.pipeline is not None

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    def test_retriever_with_custom_es_client(self, mock_resource_manager):
        """Test Retriever with custom ES client."""
        custom_es_client = Mock(spec=EsClient)
        retriever = Retriever(locale="jp", es_client=custom_es_client)

        assert retriever.es_client == custom_es_client

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_retriever_with_synonym_dict(self, mock_es_client, mock_resource_manager):
        """Test Retriever with synonym dictionary."""
        synonym_dict = Mock(spec=SynonymDict)
        retriever = Retriever(locale="jp", synonym_dict=synonym_dict)

        assert retriever.locale == "jp"
        # Should have synonym processor in the chain

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    @patch("amazon_product_search.retrieval.retriever.RetrievalPipeline")
    def test_search_backward_compatibility(self, mock_pipeline, mock_es_client, mock_resource_manager):
        """Test that search method maintains backward compatibility."""
        # Mock pipeline response
        mock_pipeline_response = Mock()
        mock_pipeline_response.results = [
            Result(product={"product_id": "123", "title": "Test"}, score=0.9)
        ]
        mock_pipeline_response.total_hits = 1

        mock_pipeline.return_value.search.return_value = mock_pipeline_response

        retriever = Retriever(locale="jp")
        response = retriever.search(
            index_name="products_jp",
            query="wireless headphones",
            fields=["product_title", "title_vector"],
            enable_synonym_expansion=True,
            lexical_boost=0.7,
            semantic_boost=1.3,
            size=20
        )

        # Verify pipeline was called
        mock_pipeline.return_value.search.assert_called_once()

        # Verify response format
        assert isinstance(response, Response)
        assert len(response.results) == 1
        assert response.total_hits == 1

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_add_retrieval_engine(self, mock_es_client, mock_resource_manager):
        """Test adding retrieval engines dynamically."""
        retriever = Retriever(locale="jp")
        initial_engine_count = len(retriever.retrieval_engines)

        mock_engine = Mock()
        retriever.add_retrieval_engine(mock_engine)

        assert len(retriever.retrieval_engines) == initial_engine_count + 1
        assert mock_engine in retriever.retrieval_engines

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_add_post_processor(self, mock_es_client, mock_resource_manager):
        """Test adding post processors."""
        retriever = Retriever(locale="jp")
        mock_processor = Mock()

        retriever.add_post_processor(mock_processor)

        # Verify processor was added to pipeline
        retriever.pipeline.add_post_processor.assert_called_once_with(mock_processor)

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_get_pipeline_info(self, mock_es_client, mock_resource_manager):
        """Test getting pipeline information."""
        retriever = Retriever(locale="jp")

        # Mock pipeline info
        retriever.pipeline.get_pipeline_info = Mock(return_value={
            "query_processor": "BaseQueryProcessor",
            "retrieval_engines": 2,
            "post_processors": 0
        })

        info = retriever.get_pipeline_info()

        assert isinstance(info, dict)
        assert info["architecture"] == "modular"
        assert info["locale"] == "jp"
        assert "query_processor" in info
        assert "retrieval_engines" in info

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_field_type_detection(self, mock_es_client, mock_resource_manager):
        """Test automatic field type detection in search method."""
        # Mock pipeline to capture the config passed to it
        captured_config = None

        def capture_config(*args, **kwargs):
            nonlocal captured_config
            captured_config = args[1]  # RetrievalConfig is second argument
            mock_response = Mock()
            mock_response.results = []
            mock_response.total_hits = 0
            return mock_response

        retriever = Retriever(locale="jp")
        retriever.pipeline.search = Mock(side_effect=capture_config)

        retriever.search(
            index_name="test_index",
            query="test",
            fields=["product_title", "description", "title_vector", "embedding"],
            size=10
        )

        # Verify field types were detected correctly
        assert captured_config is not None
        field_types = {field.name: field.field_type for field in captured_config.fields}

        assert field_types["product_title"] == FieldType.LEXICAL
        assert field_types["description"] == FieldType.LEXICAL
        assert field_types["title_vector"] == FieldType.SEMANTIC
        assert field_types["embedding"] == FieldType.SEMANTIC

    @patch("amazon_product_search.retrieval.retriever.SharedResourceManager")
    @patch("amazon_product_search.retrieval.retriever.EsClient")
    def test_fusion_weights_conversion(self, mock_es_client, mock_resource_manager):
        """Test that legacy boost parameters are converted to fusion weights."""
        captured_weights = None

        def capture_weights(*args, **kwargs):
            nonlocal captured_weights
            captured_weights = args[2] if len(args) > 2 else kwargs.get("fusion_weights")
            mock_response = Mock()
            mock_response.results = []
            mock_response.total_hits = 0
            return mock_response

        retriever = Retriever(locale="jp")
        retriever.pipeline.search = Mock(side_effect=capture_weights)

        retriever.search(
            index_name="test_index",
            query="test",
            fields=["title"],
            lexical_boost=0.8,
            semantic_boost=1.2,
            size=10
        )

        # Verify fusion weights were set correctly
        assert captured_weights is not None
        assert captured_weights["lexical"] == 0.8
        assert captured_weights["semantic"] == 1.2
