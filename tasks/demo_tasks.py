from invoke import task


@task
def eda(c):
    c.run("poetry run streamlit run src/demo/apps/eda/📊_Product_Catalogue.py")


@task
def features(c):
    c.run("poetry run streamlit run src/demo/apps/features/🤖_Tokenization.py")


@task
def es(c):
    c.run("poetry run streamlit run src/demo/apps/es/🔍_Retrieval.py")


@task
def vespa(c):
    c.run("poetry run streamlit run src/demo/apps/vespa/🔍_Retrieval.py")
