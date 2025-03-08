from collections.abc import Callable
from typing import Any

from kfp import dsl
from kfp.dsl import Metrics, Output, PipelineTask


def build_fine_tune_func(image: str) -> Callable[[str], PipelineTask]:
    @dsl.component(base_image=image)
    def fine_tune(
        project_dir: str,
        input_filename: str,
        bert_model_name: str,
        max_epochs: int,
        debug: bool,
        metrics_output: Output[Metrics],
    ) -> None:
        from collections import defaultdict

        from training.fine_tuning_cl.components import run

        metrics: list[dict[str, Any]] = run(
            project_dir,
            input_filename,
            bert_model_name,
            max_epochs=max_epochs,
            debug=debug,
        )
        # metrics are given as follows:
        # [
        #     {"epoch": 0, "metric_name": "val_spearman_cosine", "value": 0.8},
        #     {"epoch": 1, "metric_name": "val_spearman_cosine", "value": 0.9},
        # ]
        # In order to log metrics in the format of key-value pairs,
        # the metric list provided above will be converted as follows
        # {
        #     "val_spearman_cosine": [0.8, 0.9],
        # }
        metric_dict = defaultdict(list)
        for row in sorted(metrics, key=lambda metric: metric["epoch"]):
            metric_name = row["metric_name"]
            value = row["value"]
            metric_dict[metric_name].append(value)
        for key, values in metric_dict.items():
            metrics_output.log_metric(key, values[-1])
    return fine_tune
