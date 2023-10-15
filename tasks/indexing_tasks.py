from invoke import task

from amazon_product_search.constants import PROJECT_ID, PROJECT_NAME, REGION


@task
def run(
    c,
    index_name="",
    locale="jp",
    source="file",
    dest="stdout",
    dest_host="",
    extract_keywords=False,
    encode_text=False,
    nrows=None,
    runner="DirectRunner",
):
    """A task to run indexing pipeline.

    An example sequence of commands to index products from File => BigQuery => Elasticsearch:

    ```
    # File => BigQuery
    poetry run inv indexing.run \
      --locale=us \
      --encode-text \
      --nrows=10 \
      --dest=bq
    ```

    Ensure that an index is created in Elasticsearch before running the following command.
    ```
    poetry run inv es.recreate-index --index-name=products_us
    ```

    ```
    # BigQuery => Elasticsearch
    poetry run inv indexing.run \
      --locale=us \
      --source=bq \
      --dest=es \
      --dest-host=http://localhost:9200 \
      --index-name=products_us
    ```
    """
    command = [
        "poetry run python src/amazon_product_search/indexing/pipeline.py",
        f"--runner={runner}",
        f"--locale={locale}",
        f"--source={source}",
        f"--dest={dest}",
        f"--dest_host={dest_host}",
        f"--index_name={index_name}",
    ]

    if extract_keywords:
        command.append("--extract_keywords")

    if encode_text:
        command.append("--encode_text")

    if runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
        ]
    elif runner == "DataflowRunner":
        command += [
            "--num_workers=1",
            "--worker_machine_type=n2-highmem-8",
            "--sdk_location=container",
            f"--sdk_container_image=gcr.io/{PROJECT_ID}/{PROJECT_NAME}/indexing",
            f"--worker_zone={REGION}-c",
        ]

    if (runner == "DataflowRunner") or (source == "bq") or (dest == "bq"):
        command += [
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--temp_location=gs://{PROJECT_NAME}/temp",
            f"--staging_location=gs://{PROJECT_NAME}/staging",
        ]

    if nrows:
        command.append(f"--nrows={int(nrows)}")

    c.run(" ".join(command))
