from invoke import Collection, task

from tasks import (
    data_tasks,
    demo_tasks,
    es_tasks,
    indexing_tasks,
    model_tasks,
    synonyms_tasks,
    training_tasks,
    vespa_tasks,
)


@task
def lint(c):
    print("Running ruff check...")
    c.run("poetry run ruff check --fix")
    print("Running ruff format...")
    c.run("poetry run ruff format")
    print("Running mypy...")
    c.run("poetry run mypy src/amazon_product_search --explicit-package-bases  --namespace-packages")


@task(pre=[lint])
def verify(c):
    print("Running pytest...")
    c.run("poetry run pytest tests/unit")


ns = Collection()
ns.add_task(lint)
ns.add_task(verify)
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(demo_tasks, name="demo"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
ns.add_collection(Collection.from_module(indexing_tasks, name="indexing"))
ns.add_collection(Collection.from_module(model_tasks, name="model"))
ns.add_collection(Collection.from_module(synonyms_tasks, name="synonyms"))
ns.add_collection(Collection.from_module(training_tasks, name="training"))
ns.add_collection(Collection.from_module(vespa_tasks, name="vespa"))
