from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler

from amazon_product_search.constants import (
    PROJECT_ID,
    REGION,
    SERVICE_ACCOUNT,
    STAGING_BUCKET,
    VERTEX_DIR,
)
from amazon_product_search.timestamp import get_unix_timestamp


@dsl.component
def train() -> None:
    print("Train")


@dsl.component
def evaluate() -> None:
    print("Evaluate")


@dsl.component
def predict() -> None:
    print("Predict")


@dsl.pipeline(
    name="train",
)
def training_pipeline() -> None:
    train_task = train()

    evaluate_task = evaluate()
    evaluate_task.after(train_task)

    predict_task = predict()
    predict_task.after(evaluate_task)


def main():
    experiment = f"training-{get_unix_timestamp()}"
    package_path = f"{VERTEX_DIR}/training_pipeline.yaml"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
        experiment=experiment,
    )

    Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=package_path,
        type_check=True,
    )

    job = aiplatform.PipelineJob(
        display_name=experiment,
        template_path=package_path,
    )
    job.submit(
        service_account=SERVICE_ACCOUNT,
        experiment=experiment,
    )
    job._block_until_complete()


if __name__ == "__main__":
    main()