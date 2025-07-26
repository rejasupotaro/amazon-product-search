from unittest.mock import Mock, patch

import pytest

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.retrieval.core.types import FieldType, SearchField
from amazon_product_search.retrieval.factory import (
    create_basic_retrieval_pipeline,
    create_retrieval_system,
    create_retriever,
    create_search_fields,
)
from amazon_product_search.retrieval.pipeline import RetrievalPipeline
from amazon_product_search.retrieval.retriever import Retriever
from amazon_product_search.synonyms.synonym_dict import SynonymDict


class TestCreateSearchFields:
    """Test cases for create_search_fields factory function."""

    def test_create_basic_fields(self):
        """Test creating basic search fields."""
        field_names = ["title", "description", "title_vector"]
        fields = create_search_fields(field_names)

        assert len(fields) == 3
        assert fields[0].name == "title"
        assert fields[0].field_type == FieldType.LEXICAL
        assert fields[1].name == "description"
        assert fields[1].field_type == FieldType.LEXICAL
        assert fields[2].name == "title_vector"
        assert fields[2].field_type == FieldType.SEMANTIC

    def test_create_fields_with_weights(self):
        """Test creating fields with custom weights."""
        field_configs = [
            {"name": "title", "weight": 2.0},
            {"name": "description", "weight": 1.0},
            {"name": "title_vector", "weight": 1.5}
        ]
        fields = create_search_fields(field_configs)

        assert len(fields) == 3
        assert fields[0].weight == 2.0
        assert fields[1].weight == 1.0
        assert fields[2].weight == 1.5

    def test_create_fields_mixed_format(self):
        """Test creating fields with mixed string and dict formats."""
        field_configs = [
            "title",
            {"name": "description", "weight": 1.5},
            "title_vector"
        ]
        fields = create_search_fields(field_configs)

        assert len(fields) == 3
        assert fields[0].name == "title"
        assert fields[0].weight == 1.0  # Default weight
        assert fields[1].name == "description"
        assert fields[1].weight == 1.5
        assert fields[2].name == "title_vector"

    def test_empty_field_list(self):
        """Test handling empty field list."""
        fields = create_search_fields([])
        assert len(fields) == 0


class TestCreateBasicRetrievalPipeline:
    """Test cases for create_basic_retrieval_pipeline factory function."""

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_basic_pipeline(self, mock_es_client, mock_resource_manager):
        """Test creating basic retrieval pipeline."""
        pipeline = create_basic_retrieval_pipeline(locale="jp")

        assert isinstance(pipeline, RetrievalPipeline)
        assert len(pipeline.retrieval_engines) == 2  # Lexical and semantic
        assert pipeline.query_processor is not None
        assert pipeline.result_fuser is not None

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    def test_create_pipeline_with_custom_es_client(self, mock_resource_manager):
        """Test creating pipeline with custom ES client."""
        custom_es_client = Mock(spec=EsClient)
        pipeline = create_basic_retrieval_pipeline(
            locale="us",
            es_client=custom_es_client
        )

        assert isinstance(pipeline, RetrievalPipeline)
        # Verify custom ES client is used (would need to check engine internals)

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_pipeline_with_synonym_dict(self, mock_es_client, mock_resource_manager):
        """Test creating pipeline with synonym dictionary."""
        synonym_dict = Mock(spec=SynonymDict)
        pipeline = create_basic_retrieval_pipeline(
            locale="jp",
            synonym_dict=synonym_dict
        )

        assert isinstance(pipeline, RetrievalPipeline)
        # Pipeline should have synonym processor in the chain

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_pipeline_with_fusion_method(self, mock_es_client, mock_resource_manager):
        """Test creating pipeline with different fusion methods."""
        fusion_methods = ["weighted_sum", "rrf", "borda_count"]

        for method in fusion_methods:
            pipeline = create_basic_retrieval_pipeline(
                locale="jp",
                fusion_method=method
            )
            assert isinstance(pipeline, RetrievalPipeline)


class TestCreateRetriever:
    """Test cases for create_retriever factory function."""

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_basic_retriever(self, mock_es_client, mock_resource_manager):
        """Test creating basic retriever."""
        retriever = create_retriever(locale="jp")

        assert isinstance(retriever, Retriever)
        assert retriever.locale == "jp"
        assert retriever.es_client is not None
        assert retriever.resource_manager is not None

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    def test_create_retriever_with_custom_es_client(self, mock_resource_manager):
        """Test creating retriever with custom ES client."""
        custom_es_client = Mock(spec=EsClient)
        retriever = create_retriever(
            locale="us",
            es_client=custom_es_client
        )

        assert isinstance(retriever, Retriever)
        assert retriever.es_client == custom_es_client

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_retriever_with_synonym_dict(self, mock_es_client, mock_resource_manager):
        """Test creating retriever with synonym dictionary."""
        synonym_dict = Mock(spec=SynonymDict)
        retriever = create_retriever(
            locale="jp",
            synonym_dict=synonym_dict
        )

        assert isinstance(retriever, Retriever)

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_retriever_with_reranking_enabled(self, mock_es_client, mock_resource_manager):
        """Test creating retriever with reranking enabled."""
        retriever = create_retriever(
            locale="jp",
            enable_reranking=True
        )

        assert isinstance(retriever, Retriever)
        # Should have reranking post-processor added

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_retriever_with_reranking_disabled(self, mock_es_client, mock_resource_manager):
        """Test creating retriever with reranking disabled."""
        retriever = create_retriever(
            locale="jp",
            enable_reranking=False
        )

        assert isinstance(retriever, Retriever)
        # Should not have reranking post-processor


class TestCreateRetrievalSystem:
    """Test cases for create_retrieval_system factory function."""

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_basic_system(self, mock_es_client, mock_resource_manager):
        """Test creating basic retrieval system."""
        system = create_retrieval_system("basic", locale="jp")

        assert isinstance(system, RetrievalPipeline)

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_hybrid_rrf_system(self, mock_es_client, mock_resource_manager):
        """Test creating hybrid RRF system."""
        system = create_retrieval_system("hybrid_rrf", locale="jp")

        assert isinstance(system, RetrievalPipeline)
        # Should use RRF fusion method

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_advanced_system(self, mock_es_client, mock_resource_manager):
        """Test creating advanced system with all features."""
        system = create_retrieval_system("colbert_advanced", locale="jp")

        assert isinstance(system, RetrievalPipeline)
        # Should have reranking and filtering enabled

    def test_invalid_system_type(self):
        """Test handling invalid system type."""
        with pytest.raises(ValueError, match="Unknown config"):
            create_retrieval_system("invalid_type", locale="jp")

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_create_system_with_custom_config(self, mock_es_client, mock_resource_manager):
        """Test creating system with custom configuration."""
        custom_es_client = Mock(spec=EsClient)
        synonym_dict = Mock(spec=SynonymDict)

        system = create_retrieval_system(
            "basic",
            locale="jp",
            es_client=custom_es_client,
            synonym_dict=synonym_dict
        )

        assert isinstance(system, RetrievalPipeline)


class TestFactoryIntegration:
    """Integration tests for factory functions."""

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_retriever_can_perform_search(self, mock_es_client, mock_resource_manager):
        """Test that factory-created retriever can perform searches."""
        # Mock ES client response
        mock_response = Mock()
        mock_response.results = []
        mock_response.total_hits = 0
        mock_es_client.return_value.search.return_value = mock_response

        retriever = create_retriever(locale="jp")

        # This should not raise an exception
        response = retriever.search(
            index_name="test_index",
            query="test query",
            fields=["title", "title_vector"],
            size=10
        )

        assert response is not None

    @patch("amazon_product_search.retrieval.factory.SharedResourceManager")
    @patch("amazon_product_search.retrieval.factory.EsClient")
    def test_pipeline_can_perform_search(self, mock_es_client, mock_resource_manager):
        """Test that factory-created pipeline can perform searches."""
        from amazon_product_search.retrieval.core.types import FieldType, RetrievalConfig

        # Mock ES client response
        mock_response = Mock()
        mock_response.results = []
        mock_response.total_hits = 0
        mock_es_client.return_value.search.return_value = mock_response

        pipeline = create_basic_retrieval_pipeline(locale="jp")

        config = RetrievalConfig(
            index_name="test_index",
            fields=[
                SearchField(name="title", field_type=FieldType.LEXICAL),
                SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
            ],
            size=10
        )

        # This should not raise an exception
        response = pipeline.search("test query", config)

        assert response is not None

    def test_create_search_fields_type_detection(self):
        """Test automatic field type detection in create_search_fields."""
        fields = create_search_fields([
            "product_title",
            "product_description",
            "title_vector",
            "description_embedding",
            "dense_vector"
        ])

        # Non-vector fields should be lexical
        assert fields[0].field_type == FieldType.LEXICAL
        assert fields[1].field_type == FieldType.LEXICAL

        # Vector fields should be semantic
        assert fields[2].field_type == FieldType.SEMANTIC
        assert fields[3].field_type == FieldType.SEMANTIC
        assert fields[4].field_type == FieldType.SEMANTIC
