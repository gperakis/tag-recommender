import re
import unicodedata

import emoji
from tqdm import tqdm


def remove_emojis(text):
    return emoji.replace_emoji(text, "")


def to_snake_case_boosted(text: str) -> str:
    """
    Convert a string to snake_case, supporting Unicode characters,
    handling punctuation, emoticons, and other special cases related to hashtags.

    Parameters
    ----------
    text : str
        The input string to convert.

    Returns
    -------
    str
        The input string converted to snake_case.
    """
    # Normalize Unicode characters to NFKD form
    text = unicodedata.normalize("NFKD", text)

    # Remove accents from characters
    text = "".join([c for c in text if not unicodedata.combining(c)])

    # Remove emojis using the emoji library
    text = remove_emojis(text)

    # Remove punctuation marks, keeping underscores if needed
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # Replace spaces and hyphens with underscores
    text = re.sub(r"[\s\-]+", "_", text, flags=re.UNICODE)

    # Handle CamelCase or PascalCase
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z0-9])", "_", text)

    # Convert to lowercase
    text = text.lower()

    # Remove multiple underscores
    text = re.sub(r"_+", "_", text)

    # Strip leading and trailing underscores and whitespace
    return text.strip("_").strip()


def split_tags(tags_str: str) -> list[str]:
    """
    Splits a string of tags into a list.

    Parameters
    ----------
    tags_str : str

    Returns
    -------
    list[str]
        A list of individual tags.
    """
    if tags_str:
        return [tag.strip() for tag in tags_str.split(",") if tag.strip()]
    return []


def to_snake_case(text: str) -> str:
    """
    Convert a string to snake_case.

    Parameters
    ----------
    text : str
        The input string to convert.

    Returns
    -------
    str
        The input string converted to snake_case.
    """
    # If it's already in snake_case, return it
    if re.match(r"^[a-z0-9_]+$", text):
        return text

    # Handle PascalCase or camelCase
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()

    # Handle spaces, dashes, and multiple underscores
    text = re.sub(r"[\s\-]+", "_", text)

    # Remove any multiple underscores that may occur
    text = re.sub(r"_+", "_", text)

    # Strip leading or trailing underscores if present
    return text.strip("_").strip()


def normalize_hashtags(text: str) -> list[str]:
    """
    Normalize a row of hashtags by converting them to snake_case.

    Parameters
    ----------
    text : str
        The input string containing hashtags separated by commas.

    Returns
    -------
    list[str]
        The list of normalized hashtags in snake_case.
    """
    if not text:
        return []

    tags = text.split(",")

    return [to_snake_case_boosted(hashtag) for hashtag in tags]


def preprocess_corpus(corpus: list[str]) -> list[list[str]]:
    """
    Preprocess the corpus by normalizing hashtags

    Parameters
    ----------
    corpus : list[str]
        The input corpus where each entry is a comma-separated list of hashtags.

    Returns
    -------
    list[list[str]]
        The processed corpus where each entry is a list of normalized hashtags.
    """
    return [normalize_hashtags(entry) for entry in tqdm(corpus)]
