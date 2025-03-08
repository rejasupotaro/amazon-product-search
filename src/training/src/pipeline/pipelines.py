import logging
from abc import ABC, abstractmethod
from typing import Any, Type

from kfp import dsl
from kfp.compiler import Compiler
from pipeline.components.dummy import build_evaluate_func, build_preprocess_func, build_train_func

logger = logging.getLogger(__name__)



class BasePipeline(ABC):
    def __init__(self, image: str) -> None:
        self.pipeline_func = self.build_pipeline_func(image=image)

    @abstractmethod
    def build_pipeline_func(self, image: str) -> Any:
        raise NotImplementedError

    def compile(self, template_path: str) -> None:
        """Compile a pipeline function as a pipeline template."""
        compiler = Compiler()
        compiler.compile(
            pipeline_func=self.pipeline_func,
            package_path=template_path,
        )


class DummyPipeline(BasePipeline):
    def build_pipeline_func(self, image: str) -> Any:
        @dsl.pipeline(
            name="dummy",
        )
        def pipeline_func() -> None:
            preprocess_task = build_preprocess_func(image=image)(message="Hello World")

            train_task = build_train_func(image=image)()
            train_task.after(preprocess_task)

            evaluate_task = build_evaluate_func(image=image)()
            evaluate_task.after(train_task)

        return pipeline_func


PIPELINE_DICT: dict[str, Type[BasePipeline]] = {
    "dummy": DummyPipeline,
}
