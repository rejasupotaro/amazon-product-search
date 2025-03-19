import contextlib

import typer
from elasticsearch import NotFoundError
from typing_extensions import Annotated

from amazon_product_search.es.es_client import EsClient

app = typer.Typer()


@app.command()
def greet():
    print("Hello, World!")



@app.command()
def create_index(
    index_name: Annotated[str, typer.Option()],
    locale: Annotated[str, typer.Option()],
) -> None:
    es_client = EsClient()
    with contextlib.suppress(NotFoundError):
        es_client.delete_index(index_name)

    es_client.create_index(locale, index_name)
    print(f"{index_name} was created")
