from unittest.mock import Mock, patch

import pytest

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.types import (
    FieldType,
    ProcessedQuery,
    RetrievalConfig,
    RetrievalResponse,
    SearchField,
)
from amazon_product_search.retrieval.engines.lexical import LexicalRetrievalEngine
from amazon_product_search.retrieval.engines.semantic import SemanticRetrievalEngine
from amazon_product_search.retrieval.response import Result


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    client = Mock(spec=EsClient)
    client.search.return_value = Mock(
        results=[
            Result(
                product={"product_id": "123", "title": "Test Product"},
                score=0.85
            )
        ],
        total_hits=1
    )
    return client


@pytest.fixture
def lexical_query():
    """Sample processed query for lexical search."""
    return ProcessedQuery(
        raw="wireless headphones",
        normalized="wireless headphones",
        tokens=["wireless", "headphones"],
        synonyms=["bluetooth headphones", "cordless headphones"],
        metadata={"tokenized": True}
    )


@pytest.fixture
def semantic_query():
    """Sample processed query for semantic search."""
    return ProcessedQuery(
        raw="wireless headphones",
        normalized="wireless headphones",
        tokens=["wireless", "headphones"],
        vector=[0.1, 0.2, 0.3, 0.4],
        metadata={"encoded": True}
    )


@pytest.fixture
def lexical_config():
    """Retrieval configuration for lexical search."""
    return RetrievalConfig(
        index_name="products_jp",
        fields=[
            SearchField(name="product_title", field_type=FieldType.LEXICAL, weight=2.0),
            SearchField(name="product_description", field_type=FieldType.LEXICAL, weight=1.0)
        ],
        size=20,
        window_size=50,
        enable_synonyms=True
    )


@pytest.fixture
def semantic_config():
    """Retrieval configuration for semantic search."""
    return RetrievalConfig(
        index_name="products_jp",
        fields=[
            SearchField(name="title_vector", field_type=FieldType.SEMANTIC, weight=1.0)
        ],
        size=20,
        window_size=50
    )


class TestLexicalRetrievalEngine:
    """Test cases for LexicalRetrievalEngine."""

    def test_engine_initialization(self, mock_es_client):
        """Test lexical engine initialization."""
        engine = LexicalRetrievalEngine(es_client=mock_es_client)

        assert engine.engine_name == "lexical"
        assert engine.es_client == mock_es_client
        assert engine.template_loader is not None

    @patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader")
    def test_retrieve_basic_flow(self, mock_template_loader, mock_es_client,
                                lexical_query, lexical_config):
        """Test basic lexical retrieval flow."""
        # Mock template loader
        mock_template = Mock()
        mock_template.render.return_value = '{"query": {"match": {"title": "test"}}}'
        mock_template_loader.return_value.load.return_value = mock_template

        engine = LexicalRetrievalEngine(es_client=mock_es_client)
        response = engine.retrieve(lexical_query, lexical_config)

        # Verify ES search was called
        mock_es_client.search.assert_called_once()
        call_args = mock_es_client.search.call_args
        assert call_args[1]["index_name"] == "products_jp"
        assert call_args[1]["size"] == 50  # window_size

        # Verify response structure
        assert isinstance(response, RetrievalResponse)
        assert response.engine_name == "lexical"
        assert response.processing_time_ms > 0

    def test_retrieve_with_synonyms(self, mock_es_client, lexical_query, lexical_config):
        """Test lexical retrieval with synonym expansion."""
        with patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader") as mock_loader:
            mock_template = Mock()
            mock_template.render.return_value = '{"query": {"match": {"title": "test"}}}'
            mock_loader.return_value.load.return_value = mock_template

            engine = LexicalRetrievalEngine(es_client=mock_es_client)
            engine.retrieve(lexical_query, lexical_config)

            # Verify template was called with synonyms
            mock_template.render.assert_called_once()
            call_args = mock_template.render.call_args[1]
            assert "queries" in call_args
            queries = call_args["queries"]
            assert len(queries) >= 1  # Original query + synonyms

    def test_retrieve_no_lexical_fields(self, mock_es_client, lexical_query):
        """Test retrieval when no lexical fields are configured."""
        config = RetrievalConfig(
            index_name="products_jp",
            fields=[
                SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
            ],
            size=20
        )

        engine = LexicalRetrievalEngine(es_client=mock_es_client)
        response = engine.retrieve(lexical_query, config)

        # Should return empty response
        assert len(response.results) == 0
        assert response.total_hits == 0
        assert response.engine_name == "lexical"

    def test_supports_fields(self, mock_es_client):
        """Test field support detection."""
        engine = LexicalRetrievalEngine(es_client=mock_es_client)

        # Should support non-vector fields
        assert engine.supports_fields(["title", "description"])
        assert engine.supports_fields(["product_title", "product_brand"])

        # Should not support vector fields
        assert not engine.supports_fields(["title_vector", "embedding"])
        assert not engine.supports_fields(["title", "title_vector"])  # Mixed case

    @patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader")
    def test_build_lexical_query_with_filters(self, mock_template_loader, mock_es_client,
                                             lexical_query, lexical_config):
        """Test query building with filters."""
        lexical_config.filters = {"category": "electronics", "brand": ["sony", "apple"]}

        mock_template = Mock()
        mock_template.render.return_value = '{"query": {"match": {"title": "test"}}}'
        mock_template_loader.return_value.load.return_value = mock_template

        engine = LexicalRetrievalEngine(es_client=mock_es_client)
        engine.retrieve(lexical_query, lexical_config)

        # Verify ES query includes filters
        mock_es_client.search.assert_called_once()
        call_args = mock_es_client.search.call_args[1]
        assert "query" in call_args

    def test_field_weight_handling(self, mock_es_client, lexical_query, lexical_config):
        """Test that field weights are properly handled."""
        with patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader") as mock_loader:
            mock_template = Mock()
            mock_template.render.return_value = '{"query": {"match": {"title": "test"}}}'
            mock_loader.return_value.load.return_value = mock_template

            engine = LexicalRetrievalEngine(es_client=mock_es_client)
            engine.retrieve(lexical_query, lexical_config)

            # Verify fields with weights were passed to template
            mock_template.render.assert_called_once()
            call_args = mock_template.render.call_args[1]
            assert "fields" in call_args
            fields = call_args["fields"]
            assert "product_title^2.0" in fields
            assert "product_description^1.0" in fields


class TestSemanticRetrievalEngine:
    """Test cases for SemanticRetrievalEngine."""

    def test_engine_initialization(self, mock_es_client):
        """Test semantic engine initialization."""
        engine = SemanticRetrievalEngine(es_client=mock_es_client)

        assert engine.engine_name == "semantic"
        assert engine.es_client == mock_es_client
        assert engine.template_loader is not None

    @patch("amazon_product_search.retrieval.engines.semantic.TemplateLoader")
    def test_retrieve_basic_flow(self, mock_template_loader, mock_es_client,
                                semantic_query, semantic_config):
        """Test basic semantic retrieval flow."""
        # Mock template loader
        mock_template = Mock()
        mock_template.render.return_value = '{"knn": {"field": "title_vector", "query_vector": [0.1, 0.2]}}'
        mock_template_loader.return_value.load.return_value = mock_template

        engine = SemanticRetrievalEngine(es_client=mock_es_client)
        response = engine.retrieve(semantic_query, semantic_config)

        # Verify ES search was called with KNN query
        mock_es_client.search.assert_called_once()
        call_args = mock_es_client.search.call_args[1]
        assert call_args["index_name"] == "products_jp"
        assert "knn_query" in call_args

        # Verify response structure
        assert isinstance(response, RetrievalResponse)
        assert response.engine_name == "semantic"
        assert "vector_dim" in response.metadata

    def test_retrieve_no_query_vector(self, mock_es_client, semantic_config):
        """Test retrieval when query has no vector."""
        query_no_vector = ProcessedQuery(
            raw="test query",
            normalized="test query",
            tokens=["test", "query"]
            # No vector field
        )

        engine = SemanticRetrievalEngine(es_client=mock_es_client)
        response = engine.retrieve(query_no_vector, semantic_config)

        # Should return empty response
        assert len(response.results) == 0
        assert response.total_hits == 0
        assert response.engine_name == "semantic"

    def test_retrieve_no_semantic_fields(self, mock_es_client, semantic_query):
        """Test retrieval when no semantic fields are configured."""
        config = RetrievalConfig(
            index_name="products_jp",
            fields=[
                SearchField(name="title", field_type=FieldType.LEXICAL)
            ],
            size=20
        )

        engine = SemanticRetrievalEngine(es_client=mock_es_client)
        response = engine.retrieve(semantic_query, config)

        # Should return empty response
        assert len(response.results) == 0
        assert response.total_hits == 0

    def test_supports_fields(self, mock_es_client):
        """Test field support detection."""
        engine = SemanticRetrievalEngine(es_client=mock_es_client)

        # Should support vector fields
        assert engine.supports_fields(["title_vector"])
        assert engine.supports_fields(["embedding", "dense_vector"])

        # Should not support non-vector fields
        assert not engine.supports_fields(["title", "description"])
        assert not engine.supports_fields(["title", "title_vector"])  # Mixed case

    @patch("amazon_product_search.retrieval.engines.semantic.TemplateLoader")
    def test_build_semantic_query_with_filters(self, mock_template_loader, mock_es_client,
                                              semantic_query, semantic_config):
        """Test semantic query building with filters."""
        semantic_config.filters = {"product_id": ["123", "456"]}

        mock_template = Mock()
        mock_template.render.return_value = '{"knn": {"field": "title_vector", "query_vector": [0.1]}}'
        mock_template_loader.return_value.load.return_value = mock_template

        engine = SemanticRetrievalEngine(es_client=mock_es_client)
        engine.retrieve(semantic_query, semantic_config)

        # Verify template was called with filters
        mock_template.render.assert_called_once()
        call_args = mock_template.render.call_args[1]
        assert "product_ids" in call_args
        assert call_args["product_ids"] == ["123", "456"]

    @patch("amazon_product_search.retrieval.engines.semantic.TemplateLoader")
    def test_num_candidates_calculation(self, mock_template_loader, mock_es_client,
                                       semantic_query, semantic_config):
        """Test that num_candidates is calculated correctly."""
        semantic_config.window_size = 100

        mock_template = Mock()
        mock_template.render.return_value = '{"knn": {"field": "title_vector"}}'
        mock_template_loader.return_value.load.return_value = mock_template

        engine = SemanticRetrievalEngine(es_client=mock_es_client)
        engine.retrieve(semantic_query, semantic_config)

        # Verify num_candidates is 2x window_size
        mock_template.render.assert_called_once()
        call_args = mock_template.render.call_args[1]
        assert call_args["num_candidates"] == 200  # 2 * window_size


class TestEngineIntegration:
    """Integration tests for retrieval engines."""

    def test_lexical_engine_response_conversion(self, mock_es_client, lexical_query, lexical_config):
        """Test that lexical engine properly converts ES response."""
        # Mock ES response
        mock_es_response = Mock()
        mock_es_response.results = [
            Result(product={"product_id": "123", "title": "Test"}, score=0.9)
        ]
        mock_es_response.total_hits = 1
        mock_es_client.search.return_value = mock_es_response

        with patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader") as mock_template_loader:
            # Mock template loader to return valid JSON
            mock_template = Mock()
            mock_template.render.return_value = '{"query": {"bool": {"must": []}}}'
            mock_template_loader.return_value.load.return_value = mock_template

            engine = LexicalRetrievalEngine(es_client=mock_es_client)
            response = engine.retrieve(lexical_query, lexical_config)

            assert len(response.results) == 1
            assert response.results[0].product["product_id"] == "123"
            assert response.total_hits == 1

    def test_semantic_engine_response_conversion(self, mock_es_client, semantic_query, semantic_config):
        """Test that semantic engine properly converts ES response."""
        # Mock ES response
        mock_es_response = Mock()
        mock_es_response.results = [
            Result(product={"product_id": "456", "title": "Semantic Test"}, score=0.95)
        ]
        mock_es_response.total_hits = 1
        mock_es_client.search.return_value = mock_es_response

        with patch("amazon_product_search.retrieval.engines.semantic.TemplateLoader") as mock_template_loader:
            # Mock template loader to return valid JSON
            mock_template = Mock()
            mock_template.render.return_value = '{"knn": {"field": "vector", "query_vector": [], "k": 20}}'
            mock_template_loader.return_value.load.return_value = mock_template

            engine = SemanticRetrievalEngine(es_client=mock_es_client)
            response = engine.retrieve(semantic_query, semantic_config)

            assert len(response.results) == 1
            assert response.results[0].product["product_id"] == "456"
            assert response.total_hits == 1

    def test_engine_error_handling(self, mock_es_client, lexical_query, lexical_config):
        """Test that engines handle ES errors gracefully."""
        # Mock ES client to raise exception
        mock_es_client.search.side_effect = Exception("ES connection error")

        with patch("amazon_product_search.retrieval.engines.lexical.TemplateLoader") as mock_template_loader:
            # Mock template loader to return valid JSON
            mock_template = Mock()
            mock_template.render.return_value = '{"query": {"bool": {"must": []}}}'
            mock_template_loader.return_value.load.return_value = mock_template

            engine = LexicalRetrievalEngine(es_client=mock_es_client)

            # Should propagate the exception
            with pytest.raises(Exception, match="ES connection error"):
                engine.retrieve(lexical_query, lexical_config)
