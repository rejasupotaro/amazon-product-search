import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from kfp import dsl
from kfp.compiler import Compiler
from omegaconf.dictconfig import DictConfig
from pipeline.components.dummy import build_evaluate_func, build_preprocess_func, build_train_func
from pipeline.components.fine_tuning import build_fine_tune_cl_func, build_fine_tune_mlm_func

logger = logging.getLogger(__name__)



class BasePipeline(ABC):
    def __init__(self, config: DictConfig) -> None:
        self.pipeline_func = self.build_pipeline_func(config=config)

    @abstractmethod
    def build_pipeline_func(self, config: DictConfig) -> Any:
        raise NotImplementedError

    def compile(self, template_path: str) -> None:
        """Compile a pipeline function as a pipeline template."""
        compiler = Compiler()
        compiler.compile(
            pipeline_func=self.pipeline_func,
            package_path=template_path,
        )


class DummyPipeline(BasePipeline):
    def build_pipeline_func(self, config: DictConfig) -> Any:
        @dsl.pipeline(name=config.pipeline_type)
        def pipeline_func(
            message: str,
        ) -> None:
            preprocess_task = build_preprocess_func(image=config.image)(message=message)

            train_task = build_train_func(image=config.image)()
            train_task.after(preprocess_task)

            evaluate_task = build_evaluate_func(image=config.image)()
            evaluate_task.after(train_task)

        return pipeline_func


class FineTuningCLPipeline(BasePipeline):
    def build_pipeline_func(self, config: DictConfig) -> Any:
        @dsl.pipeline(name=config.pipeline_type)
        def pipeline_func(
            project_dir: str,
            input_filename: str,
            bert_model_name: str,
        ) -> None:
            max_epochs = 1
            num_gpus = 1
            debug = True

            fine_tune_task = build_fine_tune_cl_func(image=config.image)(
                project_dir=project_dir,
                input_filename=input_filename,
                bert_model_name=bert_model_name,
                max_epochs=max_epochs,
                debug=debug,
            )
            if num_gpus > 0:
                fine_tune_task.set_accelerator_type("NVIDIA_TESLA_T4")
                fine_tune_task.set_accelerator_limit(num_gpus)
        return pipeline_func


class FineTuningMLMPipeline(BasePipeline):
    def build_pipeline_func(self, config: DictConfig) -> Any:
        @dsl.pipeline(name=config.pipeline_type)
        def pipeline_func(
            project_dir: str,
            max_epochs: int,
            batch_size: int,
            num_sentences: Optional[int],
        ) -> None:
            build_fine_tune_mlm_func(image=config.image)(
                project_dir=project_dir,
                max_epochs=max_epochs,
                batch_size=batch_size,
                num_sentences=num_sentences,
            )
        return pipeline_func


PIPELINE_DICT: dict[str, Type[BasePipeline]] = {
    "dummy": DummyPipeline,
    "fine_tuning_cl": FineTuningCLPipeline,
    "fine_tuning_mlm": FineTuningMLMPipeline,
}
