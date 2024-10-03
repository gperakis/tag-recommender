import logging
from pathlib import Path

import click

from tag_recommender.process.split import split_dataset

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
def split(input_file, save_dir, random_state):
    """
    Split data into train, validation, and test sets.
    """
    split_dataset(input_file=input_file, save_dir=save_dir, random_state=random_state)
