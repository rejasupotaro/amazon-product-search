from invoke import task


@task
def eda(c):
    c.run("poetry run streamlit run src/amazon_product_search/demo/apps/eda/📊_Product_Catalogue.py")


@task
def features(c):
    c.run("poetry run streamlit run src/amazon_product_search/demo/apps/features/🤖_Tokenization.py")


@task
def search(c):
    c.run("poetry run streamlit run src/amazon_product_search/demo/apps/search/🔍_Retrieval.py")


@task
def vespa(c):
    c.run("poetry run streamlit run src/amazon_product_search/demo/apps/vespa/dev_tools.py")
