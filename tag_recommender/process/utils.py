import logging

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

logger = logging.getLogger(__name__)


def bucketize_col(df: pd.DataFrame, column: str, bins: list[int]) -> pd.Series:
    """
    Bucketizes a DataFrame column based on specified bins and labels.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    column : str
        The column to bucketize.
    bins : list[int]
        Bin edges.

    Returns
    -------
    pd.Series
        A Series with bucketized values as strings.
    """
    return pd.cut(df[column], bins=bins, include_lowest=True).apply(str)


def preprocess_data(df: pd.DataFrame, split_tags_func: callable) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by cleaning and transforming data.

    Parameters
    ----------
    df : pd.DataFrame
    split_tags_func : callable
        A function to split and normalize tags.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    logger.info("Starting data preprocessing...")

    df["root_tags"] = df["root_tags"].fillna("").progress_apply(split_tags_func)
    df["tags"] = df["tags"].fillna("").progress_apply(split_tags_func)

    df["root_tags_count"] = df["root_tags"].progress_apply(len)
    df["tags_count"] = df["tags"].progress_apply(len)

    df["is_reblog"] = df["is_reblog"].fillna(0)

    df["lang_type"] = df["lang"].progress_apply(
        lambda s: "en" if s == "en_US" else "other"
    )
    df["type_bucket"] = df["type"].progress_apply(
        lambda s: s if s == "photo" else "other"
    )

    # Filter out rows where tag counts are zero
    initial_len = len(df)
    df = df[~((df["root_tags_count"] == 0) & (df["tags_count"] == 0))].reset_index(
        drop=True
    )

    logger.info(f"Filtered out {initial_len - len(df)} rows with zero tag counts.")

    # calculate the total hashtags count within each array
    hashtags_counts = (
        pd.concat([df["root_tags"], df["tags"]]).explode().dropna().value_counts()
    )
    df["root_tags_popularity"] = df["root_tags"].progress_apply(
        lambda x: sum([hashtags_counts.get(tag, 0) for tag in x])
    )
    df["tags_popularity"] = df["tags"].progress_apply(
        lambda x: sum([hashtags_counts.get(tag, 0) for tag in x])
    )

    df["total_popularity"] = df["root_tags_popularity"] + df["tags_popularity"]

    logger.info("Data preprocessing completed.")

    return df
