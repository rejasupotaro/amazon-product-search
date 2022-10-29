import time


def get_unix_timestamp() -> int:
    """Return the current unix timestamp.

    ```
    >>> from amazon_product_search.timestamp import get_unix_timestamp
    >>> get_unix_timestamp()
    1667079753
    ```
    """
    return int(time.time())
