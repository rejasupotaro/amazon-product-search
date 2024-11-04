from invoke import task

from amazon_product_search.constants import (
    INDEXING_IMAGE_URI,
    PROJECT_ID,
    PROJECT_NAME,
    REGION,
)


@task
def build_indexing(c):
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.indexing,_IMAGE={INDEXING_IMAGE_URI} \
    """
    c.run(command)


@task
def transform(
    c,
    index_name="",
    locale="jp",
    dest="stdout",
    dest_host="",
    extract_keywords=False,
    encode_text=False,
    nrows=None,
    table_id="",
    runner="DirectRunner",
):
    """A task to run doc transformation pipeline.

    An example sequence of commands to index products from File => BigQuery => Elasticsearch:

    ```
    # File => BigQuery
    poetry run inv indexing.transform \
      --locale=us \
      --encode-text \
      --nrows=10 \
      --dest=bq \
      --table-id=docs_all_minilm_v6_v2_us
    ```
    """
    command = [
        "poetry run python src/amazon_product_search/indexing/doc_pipeline.py",
        f"--runner={runner}",
        f"--locale={locale}",
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

    if (runner == "DataflowRunner") or (dest == "bq"):
        command += [
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--temp_location=gs://{PROJECT_NAME}/temp",
            f"--staging_location=gs://{PROJECT_NAME}/staging",
        ]

    if dest == "bq" and table_id:
        command += [
            f"--table_id={table_id}",
        ]

    if nrows:
        command.append(f"--nrows={int(nrows)}")

    c.run(" ".join(command))


@task
def feed(
    c,
    index_name="",
    locale="jp",
    dest="stdout",
    dest_host="",
    nrows=None,
    table_id="",
    runner="DirectRunner",
):
    """A task to run feeding pipeline.

    Ensure that an index is created in Elasticsearch before running the following command.
    ```
    poetry run inv es.recreate-index --index-name=products_us
    ```

    ```
    # BigQuery => Elasticsearch
    poetry run inv indexing.feed \
      --locale=us \
      --dest=es \
      --dest-host=http://localhost:9200 \
      --index-name=products_us \
      --table-id=docs_all_minilm_v6_v2_us
    ```

    ```
    # BigQuery => Vespa
    poetry run inv indexing.feed \
      --locale=us \
      --dest=vespa \
      --dest-host=http://localhost:8080 \
      --index-name=product \
      --table-id=docs_all_minilm_v6_v2_us
    ```
    """
    command = [
        "poetry run python src/amazon_product_search/indexing/feeding_pipeline.py",
        f"--runner={runner}",
        f"--locale={locale}",
        f"--dest={dest}",
        f"--dest_host={dest_host}",
        f"--index_name={index_name}",
        f"--table_id={table_id}",
    ]

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

    command += [
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        f"--temp_location=gs://{PROJECT_NAME}/temp",
        f"--staging_location=gs://{PROJECT_NAME}/staging",
    ]

    if nrows:
        command.append(f"--nrows={int(nrows)}")

    c.run(" ".join(command))


@task
def encode(
    c,
    index_name="",
    locale="jp",
    dest="stdout",
    nrows=None,
    table_id="",
    runner="DirectRunner",
):
    """A task to run query encoding pipeline.

    ```
    # File => BigQuery
    poetry run inv indexing.encode \
      --locale=us \
      --dest=bq \
      --nrows=10 \
      --table-id=queries_all_minilm_v6_v2_us
    ```
    """
    command = [
        "poetry run python src/amazon_product_search/indexing/query_pipeline.py",
        f"--runner={runner}",
        f"--locale={locale}",
        f"--dest={dest}",
        f"--table_id={table_id}",
    ]

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

    if (runner == "DataflowRunner") or (dest == "bq"):
        command += [
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--temp_location=gs://{PROJECT_NAME}/temp",
            f"--staging_location=gs://{PROJECT_NAME}/staging",
        ]

    if nrows:
        command.append(f"--nrows={int(nrows)}")

    c.run(" ".join(command))
