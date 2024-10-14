import logging

import click

from tag_recommender.recommend.factories import train_co_occur_model

logger = logging.getLogger(__name__)


@click.group()
def models():
    pass


@models.command()
def train_co_occurrence():
    """
    Train a co-occurrence model.
    """
    logger.info("Training co-occurrence model...")
    train_co_occur_model()
