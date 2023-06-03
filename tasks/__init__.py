from invoke import Collection, task

from tasks import (
    data_tasks,
    demo_tasks,
    es_tasks,
    gcloud_tasks,
    synonyms_tasks,
    vespa_tasks,
)


@task
def verify(c):
    print("Running black...")
    c.run("poetry run black .")
    print("Running ruff...")
    c.run("poetry run ruff . --fix")
    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Running pytest...")
    c.run("poetry run pytest tests/unit")


ns = Collection()
ns.add_task(verify)
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(demo_tasks, name="demo"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
ns.add_collection(Collection.from_module(gcloud_tasks, name="gcloud"))
ns.add_collection(Collection.from_module(synonyms_tasks, name="synonyms"))
ns.add_collection(Collection.from_module(vespa_tasks, name="vespa"))
