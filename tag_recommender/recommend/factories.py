from tag_recommender.config import model_config
from tag_recommender.infer import MultiModelInference
from tag_recommender.process.factories import create_data_splitter
from tag_recommender.recommend.association_rules.factories import load_tag_rules_model
from tag_recommender.recommend.co_occur.factories import load_co_occur_inference_model
from tag_recommender.recommend.tag2vec.factories import load_tag2vec_model
from tag_recommender.utils.evaluate import Evaluator
from tag_recommender.utils.text import normalize_hashtags, split_tags


def create_inference_engine() -> MultiModelInference:
    # Initialize the inference engine
    co_occur = load_co_occur_inference_model()
    tag2vec = load_tag2vec_model()
    tag_rules = load_tag_rules_model()
    inference_engine = MultiModelInference([co_occur, tag2vec, tag_rules])
    return inference_engine


def evaluate_inference_engine():
    inference_engine = create_inference_engine()
    splitter = create_data_splitter()
    splitter.preprocess(
        input_file=model_config.input_file, normalize=model_config.normalize
    )
    splitter.create_corpus()

    split_fun = normalize_hashtags if model_config.normalize else split_tags
    evaluator = Evaluator(split_fun)
    evaluator.calculate_retrieval_metrics(
        model=inference_engine, corpus=splitter.validation_corpus, ks=[3, 5]
    )
    # setting the datasets to None to recalculate for the test set.
    evaluator.ground_truth = None
    evaluator.ground_truth_without_oov = None
    evaluator.calculate_retrieval_metrics(
        model=inference_engine, corpus=splitter.test_corpus, ks=[3, 5]
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    evaluate_inference_engine()
