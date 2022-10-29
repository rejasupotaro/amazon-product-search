from invoke import task

from amazon_product_search import source


@task
def split_by_locale(c):
    source.split_dataset_by_locale()
