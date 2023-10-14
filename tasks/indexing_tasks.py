from invoke import task

from amazon_product_search.constants import PROJECT_ID, PROJECT_NAME, REGION


@task
def run(
    c,
    index_name,
    locale="jp",
    dest="stdout",
    dest_host="",
    extract_keywords=False,
    encode_text=False,
    nrows=None,
    runner="DirectRunner",
):
    command = [
        "poetry run python src/amazon_product_search/indexing/pipeline.py",
        f"--runner={runner}",
        f"--index_name={index_name}",
        f"--locale={locale}",
        f"--dest={dest}",
        f"--dest_host={dest_host}",
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
