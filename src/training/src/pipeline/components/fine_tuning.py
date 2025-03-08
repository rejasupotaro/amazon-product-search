from collections.abc import Callable
from typing import Any, Optional

from kfp import dsl
from kfp.dsl import Metrics, Output, PipelineTask


def build_fine_tune_cl_func(image: str) -> Callable[[str], PipelineTask]:
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


def build_fine_tune_mlm_func(image: str) -> Callable[[str], PipelineTask]:
    @dsl.component(base_image=image)
    def fine_tune(
        project_dir: str,
        max_epochs: int,
        batch_size: int,
        num_sentences: Optional[int],
        metrics_output: Output[Metrics],
    ) -> None:
        from collections import defaultdict

        from training.fine_tuning_mlm.components import run

        metrics: list[dict[str, Any]] = run(
            project_dir,
            input_filename="merged_jp.parquet",
            bert_model_name="cl-tohoku/bert-base-japanese-char-v2",
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_sentences=num_sentences,
        )
        # metrics are given as follows:
        # [
        #     {"epoch": 0, "metric_name": "train_loss", "value": 0.2},
        #     {"epoch": 1, "metric_name": "train_loss", "value": 0.1},
        #     {"epoch": 0, "metric_name": "val_loss", "value": 0.3},
        #     {"epoch": 1, "metric_name": "val_loss", "value": 0.2},
        # ]
        # In order to log metrics in the format of key-value pairs,
        # the metric list provided above will be converted as follows
        # {
        #     "train_loss": [0.2, 0.1],
        #     "val_loss": [0.3, 0.2],
        # }
        metric_dict = defaultdict(list)
        for row in sorted(metrics, key=lambda metric: metric["epoch"]):
            metric_name = row["metric_name"]
            value = row["value"]
            metric_dict[metric_name].append(value)
        for key, values in metric_dict.items():
            metrics_output.log_metric(key, values[-1])
    return fine_tune
