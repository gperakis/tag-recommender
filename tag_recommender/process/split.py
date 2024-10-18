import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tag_recommender.load.base import read_raw_dataset
from tag_recommender.process.utils import bucketize_col, preprocess_data
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

        self.df_test: pd.DataFrame | None = None
        self.df_val: pd.DataFrame | None = None
        self.df_train: pd.DataFrame | None = None

        self.train_corpus: list[list[str]] | None = None
        self.validation_corpus: list[list[str]] | None = None
        self.test_corpus: list[list[str]] | None = None

    @property
    def stratify_cols(self):
        return ["root_tags_count_bucket", "type_bucket", "lang_type", "is_reblog"]

    def stratified_split(self, df: pd.DataFrame):
        """
        Splits the DataFrame into train, validation, and test sets using stratification.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        tuple
            (df_train, df_val, df_test)
        """
        logger.info("Starting stratified split...")
        logger.info(f"Stratify columns: {self.stratify_cols}")
        logger.info(f"Train size: {self.train_size}")
        logger.info(f"Validation size: {self.val_size}")
        logger.info(f"Test size: {self.test_size}")
        logger.info(f"Random state: {self.random_state}")

        df_train, df_temp = train_test_split(
            df,
            stratify=df[self.stratify_cols],
            train_size=self.train_size,
            random_state=self.random_state,
        )

        val_test_ratio = self.val_size / (self.val_size + self.test_size)

        df_val, df_test = train_test_split(
            df_temp,
            stratify=df_temp[self.stratify_cols],
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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs data bucketing and splitting.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        tuple
            (df_train, df_val, df_test)
        """
        logger.info("Starting data bucketing...")

        max_tags_count = max(df["root_tags_count"].max(), df["tags_count"].max())
        root_tag_bins = [0, 3, 10, 20, max_tags_count]
        tag_bins = [0, 2, 5, 10, max_tags_count]

        df["root_tags_count_bucket"] = bucketize_col(
            df, "root_tags_count", root_tag_bins
        )
        df["tags_count_bucket"] = bucketize_col(df, "tags_count", tag_bins)

        logger.info("Data bucketing completed.")

        # Perform the split
        df_train, df_val, df_test = self.stratified_split(df)
        logger.info("Data split completed.")

        return df_train, df_val, df_test

    def preprocess(
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

        self.df_train, self.df_val, self.df_test = self.run_split(df)

        # Save datasets if save_dir is provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.df_train.to_parquet(
                save_dir / "train.parquet",
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            self.df_val.to_parquet(
                save_dir / "val.parquet",
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            self.df_test.to_parquet(
                save_dir / "test.parquet",
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

        return self.df_train, self.df_val, self.df_test

    def create_corpus(self):
        """
        Create the corpus for the train, validation, and test sets.

        Returns
        -------
        None
        """
        if self.df_train is None:
            raise ValueError(
                "The dataset has not been preprocessed. "
                "Please preprocess the dataset first."
            )

        rt = "root_tags"
        t = "tags"
        train_corpus = self.df_train[rt].tolist() + self.df_train[t].tolist()
        validation_corpus = self.df_val[rt].tolist() + self.df_val[t].tolist()
        test_corpus = self.df_test[rt].tolist() + self.df_test[rt].tolist()

        # get rid of empty lists
        self.train_corpus = [arr for arr in train_corpus if arr]
        self.validation_corpus = [arr for arr in validation_corpus if arr]
        self.test_corpus = [arr for arr in test_corpus if arr]
