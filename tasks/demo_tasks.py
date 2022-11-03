from invoke import task


@task
def dataset(c):
    c.run("poetry run streamlit run src/demo/pages/dataset/dataset.py")


@task
def keyword_extraction(c):
    c.run("poetry run streamlit run src/demo/pages/keyword_extraction/keyword_extraction.py")


@task
def search(c):
    c.run("poetry run streamlit run src/demo/pages/search/search.py")


@task
def experiment(c):
    c.run("poetry run streamlit run src/demo/pages/experiment/experiment.py")
