from invoke import Collection, task

from tasks import (
    data_tasks,
    es_tasks,
    model_tasks,
    synonyms_tasks,
    vespa_tasks,
)


@task
def verify(c):
    print("Running pytest...")
    c.run("poetry run pytest tests/unit")


ns = Collection()
ns.add_task(verify)
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
ns.add_collection(Collection.from_module(model_tasks, name="model"))
ns.add_collection(Collection.from_module(synonyms_tasks, name="synonyms"))
ns.add_collection(Collection.from_module(vespa_tasks, name="vespa"))
