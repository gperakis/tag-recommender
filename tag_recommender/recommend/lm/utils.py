import itertools
import logging
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def normalize_tag(tag: str) -> str:
    """
    Normalize a tag by removing various whitespace characters,
    including tabs, newlines, and other types of spaces.
    This function does not lowercase the text.

    Parameters
    ----------
    tag : str
        The tag to normalize.

    Returns
    -------
    str
        The normalized tag.
    """
    # Replace various types of spaces with a regular space
    tag = re.sub(r"[\t\n\r\u00A0\u2002\u2003\u2009\u200B]", " ", tag)

    # Replace multiple spaces with a single space
    tag = re.sub(r"\s+", " ", tag)

    # Strip leading and trailing spaces
    tag = tag.strip()

    return tag


def create_positive_pairs(tags_list: list[str]) -> list[tuple[str, str]]:
    """
    Create positive tag pairs from a list of tags.

    Parameters
    ----------
    tags_list : list[str]

    Returns
    -------
    list[tuple[str, str]]
    """
    if len(tags_list) < 2:
        return []

    return list(itertools.combinations(tags_list, 2))


def create_negative_samples(tags: list[str], all_tags: np.array) -> str:
    """
    Create negative tag samples for a given anchor tag.

    Parameters
    ----------
    tags: list[str]
    all_tags : np.array

    Returns
    -------
    str
    """
    # Ensure negative tag is not part of the same post as the anchor
    while True:
        negative_tag = random.choice(all_tags)
        if negative_tag not in tags:
            return negative_tag


def create_row_triplets(arr: list[str], all_tags: np.array):
    """
    Process a row of tags to create triplets.

    Parameters
    ----------
    arr : list[str]
    all_tags : np.array

    Yields
    -------
    tuple[str, str, str]
        A tuple of (anchor, positive, negative).
    """
    # the 'all_tags' contain the list of tags in the row
    positive_pairs = list(itertools.combinations(arr, 2))

    for anchor, positive in positive_pairs:
        # Remove tabs and newlines from the tags
        anchor = anchor.replace("\t", " ").replace("\n", " ").strip()
        positive = positive.replace("\t", " ").replace("\n", " ").strip()

        negative = create_negative_samples(arr, all_tags)
        negative = negative.replace("\t", " ").replace("\n", " ").strip()

        if not anchor or not positive or not negative:
            logger.warning(
                f"Empty tag found in triplet: {anchor}, {positive}, {negative}"
            )
            continue

        yield anchor, positive, negative


def create_triplets_dataset(
    tags: pd.Series, all_tags: np.array, file_path: str | None = None
):
    """
    Create a triplet dataset from a list of tags.

    This function creates a triplet dataset from a list of tags. The dataset
    is written to a file with the format: anchor, positive, negative.

    Parameters
    ----------
    tags : pd.Series
        Each row is an array of hashtags
    all_tags : np.array
        All unique tags in the dataset
    file_path : str, optional (default=None)
        The file path to write the triplets dataset. If None, the file is written to
        'data/processed/triplets.txt'.

    Returns
    -------
    str
        The file path of the triplets dataset.
    """
    if not file_path:
        file_path = "data/processed/triplets.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        for arr in tqdm(tags):
            if len(arr) < 2:
                continue

            for anchor, positive, negative in create_row_triplets(arr, all_tags):
                f.write(f"{anchor}\t{positive}\t{negative}\n")

    logger.info("Triplet Training Data created.")

    return file_path


def get_device():
    if torch.backends.mps.is_available():
        logger.info("MPS device is available. Using MPS.")
        return "mps"

    if torch.cuda.is_available():
        logger.info("GPU device is available. Using GPU.")
        return "cuda"

    logger.info("MPS or GPU devices are not available. Using CPU.")
    return "cpu"


def preprocess_tags(row: pd.Series) -> list[str]:
    """
    Preprocess tags by combining tags and root_tags for each post, normalizing hashtags,
    and returning a list of tags.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame containing tags and root_tags.

    Returns
    -------
    list[str]
        A list of normalized tags.
    """
    # Split comma-separated tags and root_tags
    tags = row["tags"].split(",")
    root_tags = row["root_tags"].split(",")

    # Combine tags and root_tags. Normalize the tags by removing any kind of multiple
    # spaces.
    # The model will learn the embeddings for the tags.
    all_tags = set()
    for tag in tags + root_tags:
        tag = normalize_tag(tag)
        if tag:
            all_tags.add(tag)

    return list(all_tags)


def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]

    # Return an empty list instead of None if batch is empty
    if len(batch) == 0:
        return []

    return batch
