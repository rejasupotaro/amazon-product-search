import logging

from amazon_product_search.indexing import pipeline
from amazon_product_search.indexing.options import IndexerOptions


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    options = IndexerOptions()
    pipeline.run(options)


if __name__ == "__main__":
    main()
