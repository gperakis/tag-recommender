import logging

import click

from tag_recommender.recommend.association_rules.factories import train_tag_rules_model
from tag_recommender.recommend.co_occur.factories import train_co_occur_model
from tag_recommender.recommend.lm.model import train_triplet_loss_tag_model
from tag_recommender.recommend.tag2vec.factories import train_tag2vec_model

logger = logging.getLogger(__name__)


@click.group()
def models():
    pass


@models.command()
def train_co_occurrence():
    """Train a co-occurrence model using the CountMinSketch approach."""
    logger.info("Training co-occurrence model...")
    train_co_occur_model()


@models.command()
def train_tag2vec():
    """Train a tag2vec model using Word2Vec approach."""
    logger.info("Training tag2vec model...")
    train_tag2vec_model()


@models.command()
def train_tag_rules():
    """Train a tag rules model using PySpark and FP-Growth."""
    logger.info("Training Tag Association Rules model...")
    train_tag_rules_model()


@models.command()
@click.option(
    "--path",
    required=True,
    help="Path to the raw dataset file. " "Default is data/full_dataset.csv",
    default="data/full_dataset.csv",
)
@click.option(
    "--sample_size",
    default=100_000,
    help="Number of samples to use from the dataset. Default is 100,000.",
)
@click.option(
    "--epochs",
    default=1,
    help="Total number of epochs to train the model. Default is 1.",
)
@click.option(
    "--model_save_path",
    default="artifacts/models/fine_tuned_model",
    help="Path to save the fine-tuned model. "
    "Default is artifacts/models/fine_tuned_model.",
)
def train_triplet_tag_model(path, sample_size, epochs, model_save_path):
    """Train a triplet loss tag model by fine-tuning."""
    logger.info("Training Triplet Tag model...")
    train_triplet_loss_tag_model(path, sample_size, epochs, model_save_path)
