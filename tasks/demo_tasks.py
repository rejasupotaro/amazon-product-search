from invoke import task


@task
def eda(c):
    c.run("poetry run streamlit run src/demo/apps/eda/📊_Product_Catalogue.py")


@task
def search(c):
    c.run("poetry run streamlit run src/demo/apps/search/🏠_Home.py")
