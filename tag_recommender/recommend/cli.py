import logging

import click

from tag_recommender.recommend.association_rules.factories import train_tag_rules_model
from tag_recommender.recommend.co_occur.factories import train_co_occur_model
from tag_recommender.recommend.tag2vec.factories import train_tag2vec_model

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


@models.command()
def train_tag2vec():
    """
    Train a tag2vec model.
    """
    logger.info("Training tag2vec model...")
    train_tag2vec_model()


@models.command()
def train_tag_rules():
    """
    Train a tag rules model.
    """
    logger.info("Training Tag Association Rules model...")
    train_tag_rules_model()
