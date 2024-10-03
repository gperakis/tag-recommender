import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_raw_dataset(
    fname: str = "full_dataset.csv", cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Read the dataset from the given file.

    Parameters
    ----------
    fname : str
        File name to read the dataset from.
    cols : list[str] | None
        Columns to read from the dataset.

    Returns
    -------
    pd.DataFrame
    """
    logger.info(f"Reading dataset from {fname}...")
    path = Path(fname)
    if not path.exists():
        logger.info(f"File not found in : {path}. Searching in data directory")
        path = Path("data") / fname

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, usecols=cols)
    df["root_tags"] = df["root_tags"].fillna("")
    df["tags"] = df["tags"].fillna("")
    logger.info("Dataset read successfully.")
    return df
