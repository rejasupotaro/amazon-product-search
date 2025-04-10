from enum import StrEnum, auto
from typing import Type

from indexing.pipelines.base import BasePipeline
from indexing.pipelines.query_pipeline import QueryPipeline


class PipelineType(StrEnum):
    TRANSFORM = auto()
    QUERY = auto()
    FEED = auto()

    @property
    def pipeline_class(self) -> Type[BasePipeline]:
        return {
            PipelineType.QUERY: QueryPipeline,
        }[self]
