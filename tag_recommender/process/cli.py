import logging
from pathlib import Path

import click

from tag_recommender.process.split import DataSplitter

logger = logging.getLogger(__name__)


@click.group()
def data():
    pass


@data.command()
@click.option(
    "--input_file",
    type=click.Path(exists=True, path_type=str),
    required=True,
    help="Path to the input CSV file.",
)
@click.option(
    "--save_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save the split datasets.",
)
@click.option(
    "--random_state", type=int, default=42, help="Random seed for reproducibility."
)
@click.option(
    "--train_size",
    type=float,
    default=0.8,
    help="Proportion of the data to include in the train split.",
)
@click.option(
    "--val_size",
    type=float,
    default=0.10,
    help="Proportion of the data to include in the validation split.",
)
@click.option(
    "--test_size",
    type=float,
    default=0.10,
    help="Proportion of the data to include in the test split.",
)
@click.option(
    "--normalize",
    type=bool,
    default=True,
    help="Whether to normalize tags during preprocessing.",
)
def split(
    input_file, save_dir, random_state, train_size, val_size, test_size, normalize
):
    """
    Split data into train, validation, and test sets.
    """
    splitter = DataSplitter(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    splitter.preprocess(input_file=input_file, save_dir=save_dir, normalize=normalize)
