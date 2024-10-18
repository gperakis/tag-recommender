from tag_recommender.config import model_config
from tag_recommender.process.factories import create_data_splitter
from tag_recommender.recommend.tag2vec.tag2vec import Tag2VecModel
from tag_recommender.utils.evaluate import Evaluator
from tag_recommender.utils.text import normalize_hashtags, split_tags


def create_tag2vec_model() -> Tag2VecModel:
    """
    Factory method to create a Tag2VecModel object.

    Returns
    -------
    Tag2VecModel
    """
    splitter = create_data_splitter()
    split_fun = normalize_hashtags if model_config.normalize else split_tags
    evaluator = Evaluator(split_fun)

    return Tag2VecModel(model_config, splitter, evaluator)


def train_tag2vec_model() -> Tag2VecModel:
    """Train the Tag2Vec model."""
    tag2vec = create_tag2vec_model()
    tag2vec.train()
    return tag2vec


def evaluate_trained_tag2vec_model() -> Tag2VecModel:
    """Evaluate the trained Tag2Vec model."""
    tag2vec = create_tag2vec_model()
    tag2vec.load_model()
    tag2vec.preprocess()
    tag2vec.create_corpus()
    tag2vec.evaluate()
    return tag2vec


def load_tag2vec_model() -> Tag2VecModel:
    """Load the trained Tag2Vec model."""
    tag2vec = create_tag2vec_model()
    tag2vec.load_model()
    return tag2vec
