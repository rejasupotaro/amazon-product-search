from enum import StrEnum, auto
from typing import Type

from indexing.pipelines.base import BasePipeline
from indexing.pipelines.doc_pipeline import DocPipeline
from indexing.pipelines.feed_pipeline import FeedPipeline
from indexing.pipelines.query_pipeline import QueryPipeline


class PipelineType(StrEnum):
    QUERY = auto()
    DOC = auto()
    FEED = auto()

    @property
    def pipeline_class(self) -> Type[BasePipeline]:
        return {
            PipelineType.QUERY: QueryPipeline,
            PipelineType.DOC: DocPipeline,
            PipelineType.FEED: FeedPipeline,
        }[self]
