from invoke import Collection, task

from tasks import data_tasks, es_tasks, gcloud_tasks, synonyms_tasks


@task
def format(c):
    """Run formatters (isort and black)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")
    print("Done")


@task
def lint(c):
    """Run linters (isort, black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort . --check")

    print("Running black...")
    c.run("poetry run black . --check")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests tasks")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")


@task
def demo(c):
    c.run("poetry run streamlit run src/demo/🏠_Home.py")


ns = Collection()
ns.add_task(format)
ns.add_task(lint)
ns.add_task(demo)
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(gcloud_tasks, name="gcloud"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
ns.add_collection(Collection.from_module(synonyms_tasks, name="synonyms"))
