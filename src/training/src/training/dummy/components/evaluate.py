from kfp import dsl
from kfp.dsl import Metrics, Output


def build_evaluate_func() -> dsl.ContainerOp:
    @dsl.container
    def evaluate(
        metrics_output: Output[Metrics],
    ) -> None:
        print("Evaluate")
        metrics_output.log_metric("loss", [0.2, 0.1])
        metrics_output.log_metric("acc", [0.8, 0.9])

    return evaluate()
