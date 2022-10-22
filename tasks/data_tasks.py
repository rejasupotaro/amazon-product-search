from invoke import task

from amazon_product_search import source


@task
def splitt_by_locale(c):
    source.split_dataset_by_locale()
