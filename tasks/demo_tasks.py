from invoke import task


@task
def eda(c):
    c.run("poetry run streamlit run src/demo/apps/eda/📊_Product_Catalogue.py")


@task
def features(c):
    c.run("poetry run streamlit run src/demo/apps/features/🤖_Tokenization.py")


@task
def search(c):
    c.run("poetry run streamlit run src/demo/apps/search/🔍_Retrieval.py")


@task
def experiment(c):
    c.run("poetry run streamlit run src/demo/apps/experiment/📊_Experiment.py")


@task
def vespa(c):
    c.run("poetry run streamlit run src/demo/apps/vespa/dev_tools.py")
