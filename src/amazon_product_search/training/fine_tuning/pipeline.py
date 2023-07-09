from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler

from amazon_product_search.constants import (
    PROJECT_ID,
    PROJECT_NAME,
    REGION,
    SERVICE_ACCOUNT,
    STAGING_BUCKET,
    TRAINING_IMAGE_URI,
    VERTEX_DIR,
)


@dsl.component(
    base_image=TRAINING_IMAGE_URI,
    install_kfp_package=False,
)
def fine_tune(data_dir: str) -> None:
    from amazon_product_search.training.fine_tuning.components import run

    run(
        data_dir,
        input_filename="merged_jp.parquet",
        bert_model_name="cl-tohoku/bert-base-japanese-char-v2",
        max_epochs=1,
    )


@dsl.pipeline(
    name="fine_tuning",
)
def pipeline_func(data_dir: str) -> None:
    fine_tune(data_dir=data_dir)


def main() -> None:
    project_dir = f"gs://{PROJECT_NAME}"
    data_dir = f"{project_dir}/data"
    pipeline_parameters = {
        "data_dir": data_dir,
    }
    experiment = "fine-tuning"
    package_path = f"{VERTEX_DIR}/fine_tuning.yaml"

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
