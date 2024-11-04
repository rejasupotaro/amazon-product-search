from invoke import task

from amazon_product_search import source


@task
def merge_and_split(c):
    source.merge_and_split()
