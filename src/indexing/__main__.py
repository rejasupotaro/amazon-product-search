import logging

from indexing import pipeline
from indexing.options import IndexerOptions


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    options = IndexerOptions()
    pipeline.run(options)


if __name__ == "__main__":
    main()
