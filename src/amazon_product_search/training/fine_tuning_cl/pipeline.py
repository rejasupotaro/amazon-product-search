from typing import Any

from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler
from kfp.dsl import Metrics, Output

from amazon_product_search.constants import (
    PROJECT_ID,
    PROJECT_NAME,
    REGION,
    SERVICE_ACCOUNT,
    STAGING_BUCKET,
    TRAINING_IMAGE_URI,
    VERTEX_DIR,
)
from amazon_product_search.core.timestamp import get_unix_timestamp


@dsl.component(
    base_image=TRAINING_IMAGE_URI,
)
def fine_tune(
    project_dir: str,
    max_epochs: int,
    metrics_output: Output[Metrics],
) -> None:
    from collections import defaultdict

    from amazon_product_search.training.fine_tuning_cl.components import run

    metrics: list[dict[str, Any]] = run(
        project_dir,
        input_filename="merged_jp.parquet",
        bert_model_name="cl-tohoku/bert-base-japanese-char-v2",
        max_epochs=max_epochs,
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


@dsl.pipeline(
    name="fine_tuning_cl",
)
def pipeline_func(project_dir: str, max_epochs: int) -> None:
    fine_tune(project_dir=project_dir, max_epochs=max_epochs)


def main() -> None:
    project_dir = f"gs://{PROJECT_NAME}"
    pipeline_parameters = {
        "project_dir": project_dir,
        "max_epochs": 1,
    }
    experiment = "fine-tuning-cl-1"
    display_name = f"fine-tuning-cl-{get_unix_timestamp()}"
    package_path = f"{VERTEX_DIR}/fine_tuning_cl.yaml"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
        experiment=experiment,
    )

    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path,
        type_check=True,
        pipeline_parameters=pipeline_parameters,
    )

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=package_path,
    )
    job.submit(
        service_account=SERVICE_ACCOUNT,
        experiment=experiment,
    )
    job._block_until_complete()


if __name__ == "__main__":
    main()
