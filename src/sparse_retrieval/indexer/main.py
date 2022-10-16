import logging

from sparse_retrieval.indexer import pipeline
from sparse_retrieval.indexer.options import IndexerOptions


def main():
    logging.getLogger().setLevel(logging.INFO)

    options = IndexerOptions()
    pipeline.run(options)


if __name__ == "__main__":
    main()
