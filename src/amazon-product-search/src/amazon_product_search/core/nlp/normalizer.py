import logging
import re
import unicodedata

TAG_PATTERN = re.compile(r"<[/a-z0-1 ]*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")
ESCAPE_JSON_PATTERN = re.compile(r"['\\\"/\b\f\n\r\t]")


def remove_html_tags(s: str) -> str:
    return TAG_PATTERN.sub(" ", s)


def remove_punctuations(s: str) -> str:
    """Replace non-letters except `#+.°` with spaces.

    Args:
        s (str): A string to process.

    Returns:
        str: The processed string.
    """
    return re.sub(r"[^#+.°\w+]", " ", s)


def remove_surrogates(s: str) -> str:
    """Remove surrogate pairs.

    This function intends to remove unusual special characters such as emojis.
    In Python, we can identify surrogate pairs by measuring the length.
    We remove characters whose length is not 2 bytes excluding BOM (Byte Order Mark) in UTF-16.

    "-be" and `-le` stand for big-endian and little-endian, respectively.
    The suffix is needed to skip BOM when measuring the length of characters.

    Args:
        s (str): A string to process.

    Returns:
        str: The processed string.
    """
    return "".join([c for c in s if len(c.encode("utf-16-be")) == 2])


def remove_extra_spaces(s: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", s).strip()


def escape_json(s: str) -> str:
    return ESCAPE_JSON_PATTERN.sub(" ", s).strip()


def normalize_doc(s: str) -> str:
    """Normalize a document.

    In this function, we perform the following steps:
    1. Remove HTML tags.
    2. Normalize unicode characters.
    3. Convert all characters to lowercase.
    4. Remove punctuations.
    5. Remove extra spaces.

    Args:
        s (str): A string to process.

    Returns:
        str: The processed string.
    """
    try:
        s = remove_html_tags(s)
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()
        s = remove_punctuations(s)
        s = remove_extra_spaces(s)
    except Exception as e:
        logging.error(f"Received an unknown exception: {e} when processing {s}")
        raise
    return s


def normalize_query(s: str) -> str:
    """Normalize a query.

    In this function, we perform the following steps:
    1. Normalize unicode characters.
    2. Convert all characters to lowercase.
    3. Remove extra spaces.

    Unlike `normalize_doc`, we do not remove HTML tags and punctuations
    because they are rarely used in queries and if they appear in queries,
    they are likely to be meaningful.

    Args:
        s (str): A string to process.

    Returns:
        str: The processed string.
    """
    try:
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()
        s = escape_json(s)
        s = remove_extra_spaces(s)
    except Exception as e:
        logging.error(f"Received an unknown exception: {e} when processing {s}")
        raise
    return s
