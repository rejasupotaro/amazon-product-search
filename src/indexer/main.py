import logging

from indexer import pipeline
from indexer.options import IndexerOptions


def main():
    logging.getLogger().setLevel(logging.INFO)

    options = IndexerOptions()
    pipeline.run(options)


if __name__ == "__main__":
    main()
