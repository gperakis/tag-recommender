from tag_recommender.config import model_config, spark_config, tag_rules_config
from tag_recommender.process.factories import create_data_splitter
from tag_recommender.recommend.association_rules.model import TagRules
from tag_recommender.utils.evaluate import Evaluator
from tag_recommender.utils.text import normalize_hashtags, split_tags


def create_tag_rules_model() -> TagRules:
    """
    Factory method to create a TagRules object.

    Returns
    -------
    TagRules
    """
    splitter = create_data_splitter()
    split_fun = normalize_hashtags if model_config.normalize else split_tags
    evaluator = Evaluator(split_fun)
    return TagRules(model_config, splitter, evaluator, spark_config, tag_rules_config)


def train_tag_rules_model() -> TagRules:
    """Train the TagRules model"""
    tag_rules = create_tag_rules_model()
    tag_rules.train()
    return tag_rules


def evaluate_trained_tag_rules_model() -> TagRules:
    """Evaluate the trained TagRules model"""
    tag_rules = create_tag_rules_model()
    tag_rules.load_model()
    tag_rules.preprocess()
    tag_rules.create_corpus()
    tag_rules.evaluate()
    return tag_rules


def load_tag_rules_model() -> TagRules:
    """Load the trained TagRules model"""
    tag_rules = create_tag_rules_model()
    tag_rules.load_model()
    return tag_rules


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    train_tag_rules_model()
    evaluate_trained_tag_rules_model()
