import logging

from pipeline.cli import app
from pipeline.settings import get_settings

settings = get_settings()


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
