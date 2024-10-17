from tag_recommender.infer import MultiModelInference
from tag_recommender.recommend.co_occur.factories import load_co_occur_inference_model
from tag_recommender.recommend.tag2vec.factories import load_tag2vec_model


def create_inference_engine() -> MultiModelInference:
    # Initialize the inference engine
    co_occur = load_co_occur_inference_model()
    tag2vec = load_tag2vec_model()
    inference_engine = MultiModelInference([co_occur, tag2vec])
    return inference_engine
