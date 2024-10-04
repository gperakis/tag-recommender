import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tag_recommender.load.base import read_raw_dataset
from tag_recommender.process.process import bucketize_col, preprocess_data
from tag_recommender.utils.general import generate_labels
from tag_recommender.utils.text import normalize_hashtags, split_tags

tqdm.pandas()

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(
        self,
        train_size: float = 0.8,
        val_size: float = 0.10,
        test_size: float = 0.10,
        random_state: int = 42,
    ):
        """
        Initialize the Splitter.

        Parameters
        ----------
        train_size : float (default: 0.7)
            Proportion of the data to include in the train split.
        val_size : float (default: 0.15)
            Proportion of the data to include in the validation split.
        test_size : float (default: 0.15)
            Proportion of the data to include in the test split.
        random_state : int (default: 42)
            Random seed for reproducibility.
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        if self.train_size + self.val_size + self.test_size != 1.0:
            raise ValueError(
                "The sum of train_size, val_size, and test_size should be 1.0."
            )

        self.stratify_cols = [
            "tags_count_bucket",
            "root_tags_count_bucket",
            "type_bucket",
        ]

    def stratified_split(self, df: pd.DataFrame, stratify_cols: list[str]):
        """
        Splits the DataFrame into train, validation, and test sets using stratification.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        stratify_cols : list[str]
            Columns to use for stratification.

        Returns
        -------
        tuple
            (df_train, df_val, df_test)
        """
        logger.info("Starting stratified split...")
        logger.info(f"Stratify columns: {stratify_cols}")
        logger.info(f"Train size: {self.train_size}")
        logger.info(f"Validation size: {self.val_size}")
        logger.info(f"Test size: {self.test_size}")
        logger.info(f"Random state: {self.random_state}")

        df_train, df_temp = train_test_split(
            df,
            stratify=df[stratify_cols],
            train_size=self.train_size,
            random_state=self.random_state,
        )

        val_test_ratio = self.val_size / (self.val_size + self.test_size)

        df_val, df_test = train_test_split(
            df_temp,
            stratify=df_temp[stratify_cols],
            test_size=val_test_ratio,
            random_state=self.random_state,
        )
        logger.info("Stratified split completed.")
        logger.info(f"Train size: {len(df_train)}")
        logger.info(f"Validation size: {len(df_val)}")
        logger.info(f"Test size: {len(df_test)}")

        return df_train, df_val, df_test

    def run_split(
        self,
        df: pd.DataFrame,
        bins: list[int] | None = None,
        labels: list[str] | None = None,
        save_dir: Path | None = None,
    ):
        """
        Performs data bucketing and splitting.

        Parameters
        ----------
        df : pd.DataFrame
        bins : list[int] | None (default: None)
            Bin edges for bucketing.
        labels : list[str] | None (default: None)
            Labels for bins. If None, labels are generated.
        save_dir : Path | None (default: None)
            Directory to save split datasets.

        Returns
        -------
        tuple
            (df_train, df_val, df_test)
        """
        logger.info("Starting data bucketing...")

        if bins is None:
            max_tags_count = max(df["root_tags_count"].max(), df["tags_count"].max())
            bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, max_tags_count + 1]

        if labels is None:
            labels = generate_labels(bins)

        df["tags_count_bucket"] = bucketize_col(df, "tags_count", bins, labels)
        df["root_tags_count_bucket"] = bucketize_col(
            df, "root_tags_count", bins, labels
        )

        logger.info("Data bucketing completed.")

        # Perform the split
        df_train, df_val, df_test = self.stratified_split(
            df, stratify_cols=self.stratify_cols
        )

        # Save datasets if save_dir is provided
        if save_dir:
            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            df_train.to_csv(save_dir / "train.csv", index=False)
            df_val.to_csv(save_dir / "val.csv", index=False)
            df_test.to_csv(save_dir / "test.csv", index=False)
            logger.info(f"Datasets saved to {save_dir}")

        return df_train, df_val, df_test

    def process(
        self,
        input_file: str,
        normalize: bool = True,
        save_dir: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main function to split data from an input file.

        Splits data into train, validation, and test sets.
        Save the datasets if a `save_dir` is provided.

        Parameters
        ----------
        input_file : str
            Path to the input CSV file.

        normalize : bool (default: True)
            Whether to normalize tags during preprocessing.

        save_dir : str | None (default: None)
            Directory to save split datasets.

        Returns
        -------
        tuple
            (df_train, df_val, df_test)
        """
        cols = ["type", "lang", "is_reblog", "tags", "root_tags"]
        df = read_raw_dataset(input_file, cols)

        split_tags_func = normalize_hashtags if normalize else split_tags

        df = preprocess_data(df, split_tags_func=split_tags_func)

        df_train, df_val, df_test = self.run_split(df, save_dir=save_dir)
        return df_train, df_val, df_test
