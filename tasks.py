from invoke import task

import misc
from indexer import pipeline


@task
def index(c):
    pipeline.run()


@task
def lint(c):
    """Run linters (isort and black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests tasks.py")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")


@task
def split_dataset_by_locale(c):
    misc.split_dataset_by_locale()
