from unittest.mock import Mock

import pytest

from amazon_product_search.retrieval.core.types import (
    FieldType,
    ProcessedQuery,
    RetrievalConfig,
    RetrievalResponse,
    SearchField,
)
from amazon_product_search.retrieval.pipeline import RetrievalPipeline
from amazon_product_search.retrieval.response import Result


@pytest.fixture
def mock_query_processor():
    """Mock query processor for testing."""
    processor = Mock()
    processor.process.return_value = ProcessedQuery(
        raw="test query",
        normalized="test query",
        tokens=["test", "query"],
        vector=[0.1, 0.2, 0.3],
        synonyms=["test"],
        metadata={"processed": True}
    )
    return processor


@pytest.fixture
def mock_retrieval_engine():
    """Mock retrieval engine for testing."""
    engine = Mock()
    engine.engine_name = "test_engine"
    engine.retrieve.return_value = RetrievalResponse(
        results=[
            Result(
                product={"product_id": "123", "title": "Test Product"},
                score=0.9
            )
        ],
        total_hits=1,
        engine_name="test_engine",
        processing_time_ms=10.0
    )
    return engine


@pytest.fixture
def mock_result_fuser():
    """Mock result fuser for testing."""
    fuser = Mock()
    fuser.fuse.return_value = RetrievalResponse(
        results=[
            Result(
                product={"product_id": "123", "title": "Test Product"},
                score=0.95
            )
        ],
        total_hits=1,
        engine_name="fused",
        processing_time_ms=15.0
    )
    return fuser


@pytest.fixture
def mock_post_processor():
    """Mock post processor for testing."""
    processor = Mock()
    processor.process.return_value = RetrievalResponse(
        results=[
            Result(
                product={"product_id": "123", "title": "Test Product"},
                score=0.98
            )
        ],
        total_hits=1,
        engine_name="processed",
        processing_time_ms=20.0
    )
    return processor


@pytest.fixture
def retrieval_config():
    """Sample retrieval configuration."""
    return RetrievalConfig(
        index_name="test_index",
        fields=[
            SearchField(name="title", field_type=FieldType.LEXICAL),
            SearchField(name="title_vector", field_type=FieldType.SEMANTIC)
        ],
        size=10,
        enable_synonyms=True
    )


class TestRetrievalPipeline:
    """Test cases for RetrievalPipeline."""

    def test_pipeline_initialization(self, mock_query_processor, mock_retrieval_engine, mock_result_fuser):
        """Test pipeline initialization with required components."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        assert pipeline.query_processor == mock_query_processor
        assert len(pipeline.retrieval_engines) == 1
        assert pipeline.result_fuser == mock_result_fuser
        assert len(pipeline.post_processors) == 0

    def test_pipeline_with_post_processors(self, mock_query_processor, mock_retrieval_engine,
                                          mock_result_fuser, mock_post_processor):
        """Test pipeline initialization with post processors."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser,
            post_processors=[mock_post_processor]
        )

        assert len(pipeline.post_processors) == 1
        assert pipeline.post_processors[0] == mock_post_processor

    def test_search_basic_flow(self, mock_query_processor, mock_retrieval_engine,
                               mock_result_fuser, retrieval_config):
        """Test basic search flow through pipeline."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        response = pipeline.search("test query", retrieval_config)

        # Verify all components were called
        mock_query_processor.process.assert_called_once_with("test query", retrieval_config)
        mock_retrieval_engine.retrieve.assert_called_once()
        mock_result_fuser.fuse.assert_called_once()

        # Verify response structure
        assert isinstance(response, RetrievalResponse)
        assert len(response.results) == 1
        assert response.results[0].product["product_id"] == "123"

    def test_search_with_post_processing(self, mock_query_processor, mock_retrieval_engine,
                                        mock_result_fuser, mock_post_processor, retrieval_config):
        """Test search flow with post processing."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser,
            post_processors=[mock_post_processor]
        )

        response = pipeline.search("test query", retrieval_config)

        # Verify post processor was called
        mock_post_processor.process.assert_called_once()

        # Verify final response is from post processor
        assert response.results[0].score == 0.98  # Score from mock_post_processor

    def test_search_with_multiple_engines(self, mock_query_processor, mock_result_fuser, retrieval_config):
        """Test search with multiple retrieval engines."""
        engine1 = Mock()
        engine1.engine_name = "engine1"
        engine1.retrieve.return_value = RetrievalResponse(
            results=[Result(product={"product_id": "1"}, score=0.8)],
            total_hits=1,
            engine_name="engine1"
        )

        engine2 = Mock()
        engine2.engine_name = "engine2"
        engine2.retrieve.return_value = RetrievalResponse(
            results=[Result(product={"product_id": "2"}, score=0.7)],
            total_hits=1,
            engine_name="engine2"
        )

        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[engine1, engine2],
            result_fuser=mock_result_fuser
        )

        pipeline.search("test query", retrieval_config)

        # Verify both engines were called
        engine1.retrieve.assert_called_once()
        engine2.retrieve.assert_called_once()

        # Verify fuser was called with results from both engines
        mock_result_fuser.fuse.assert_called_once()
        call_args = mock_result_fuser.fuse.call_args[0]
        assert len(call_args[0]) == 2  # Two retrieval responses

    def test_search_with_fusion_weights(self, mock_query_processor, mock_retrieval_engine,
                                       mock_result_fuser, retrieval_config):
        """Test search with fusion weights."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        fusion_weights = {"test_engine": 0.8}
        pipeline.search("test query", retrieval_config, fusion_weights)

        # Verify fuser was called with weights
        mock_result_fuser.fuse.assert_called_once()
        call_args = mock_result_fuser.fuse.call_args
        assert call_args[1]["fusion_weights"] == fusion_weights

    def test_add_post_processor(self, mock_query_processor, mock_retrieval_engine,
                               mock_result_fuser, mock_post_processor):
        """Test adding post processors dynamically."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        assert len(pipeline.post_processors) == 0

        pipeline.add_post_processor(mock_post_processor)
        assert len(pipeline.post_processors) == 1
        assert pipeline.post_processors[0] == mock_post_processor

    def test_remove_post_processor(self, mock_query_processor, mock_retrieval_engine,
                                  mock_result_fuser, mock_post_processor):
        """Test removing post processors."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser,
            post_processors=[mock_post_processor]
        )

        assert len(pipeline.post_processors) == 1

        pipeline.remove_post_processor(type(mock_post_processor))
        assert len(pipeline.post_processors) == 0

    def test_get_pipeline_info(self, mock_query_processor, mock_retrieval_engine, mock_result_fuser):
        """Test pipeline info retrieval."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        info = pipeline.get_pipeline_info()

        assert isinstance(info, dict)
        assert "query_processor" in info
        assert "retrieval_engines" in info
        assert "result_fuser" in info
        assert "post_processors" in info
        assert info["retrieval_engines"] == 1
        assert info["post_processors"] == 0

    def test_empty_retrieval_engines_error(self, mock_query_processor, mock_result_fuser):
        """Test that pipeline requires at least one retrieval engine."""
        with pytest.raises(ValueError, match="At least one retrieval engine is required"):
            RetrievalPipeline(
                query_processor=mock_query_processor,
                retrieval_engines=[],
                result_fuser=mock_result_fuser
            )

    def test_search_metadata_collection(self, mock_query_processor, mock_retrieval_engine,
                                       mock_result_fuser, retrieval_config):
        """Test that pipeline collects metadata from all stages."""
        pipeline = RetrievalPipeline(
            query_processor=mock_query_processor,
            retrieval_engines=[mock_retrieval_engine],
            result_fuser=mock_result_fuser
        )

        response = pipeline.search("test query", retrieval_config)

        # Verify metadata is collected
        assert "pipeline_stages" in response.metadata
        assert "query_processing" in response.metadata["pipeline_stages"]
        assert "engines_used" in response.metadata["pipeline_stages"]
        assert "fusion_applied" in response.metadata["pipeline_stages"]
