from collections.abc import Callable

from kfp import dsl
from kfp.dsl import Metrics, Output, PipelineTask
from kfp.dsl.container_component_class import ContainerComponent


def build_preprocess_func(image: str) -> Callable[[str], PipelineTask]:
    @dsl.component(base_image=image)
    def preprocess(message: str) -> None:
        import logging
        logging.info(message)
    return preprocess


def build_train_func(image: str) -> Callable[[None], ContainerComponent]:
    @dsl.container_component
    def train() -> dsl.ContainerSpec:
        return dsl.ContainerSpec(
            image=image,
            command=["python"],
            args=["-c", "print('Hello World')"],
        )
    return train


def build_evaluate_func(image: str) -> Callable[[Output[Metrics]], PipelineTask]:
    @dsl.component(base_image=image)
    def evaluate(metrics_output: Output[Metrics]) -> None:
        print("Evaluate")
        metrics_output.log_metric("loss", [0.2, 0.1])
        metrics_output.log_metric("acc", [0.8, 0.9])
    return evaluate
