import logging
import re
import unicodedata

TAG_PATTERN = re.compile(r"<[/a-z0-1 ]*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")


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


def normalize_doc(s: str) -> str:
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
    try:
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()
        s = remove_extra_spaces(s)
    except Exception as e:
        logging.error(f"Received an unknown exception: {e} when processing {s}")
        raise
    return s
