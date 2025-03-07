from collections.abc import Callable

from kfp import dsl
from kfp.dsl import Metrics, Output, PipelineTask


def build_evaluate_func(image: str) -> Callable[[Output[Metrics]], PipelineTask]:
    @dsl.component(base_image=image)
    def evaluate(metrics_output: Output[Metrics]) -> None:
        print("Evaluate")
        metrics_output.log_metric("loss", [0.2, 0.1])
        metrics_output.log_metric("acc", [0.8, 0.9])
    return evaluate
