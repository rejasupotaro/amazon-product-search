import typer
from typing_extensions import Annotated

import training.dummy.pipeline as dummy_pipeline

app = typer.Typer()


@app.command()
def compile() -> None:
    print("Compiling pipeline")


@app.command()
def run(
    project_id: Annotated[str, typer.Option()],
    region: Annotated[str, typer.Option()],
    service_account: Annotated[str, typer.Option()],
    templates_dir: Annotated[str, typer.Option()],
    staging_bucket: Annotated[str, typer.Option()],
) -> None:
    dummy_pipeline.run(project_id, region, service_account, templates_dir, staging_bucket)
