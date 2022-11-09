from invoke import task

from amazon_product_search.synonyms import generator


@task
def generate(c):
    generator.generate()
