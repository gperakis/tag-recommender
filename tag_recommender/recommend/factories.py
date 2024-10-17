from tag_recommender.infer import MultiModelInference
from tag_recommender.recommend.co_occur.factories import load_co_occur_inference_model


def create_inference_engine() -> MultiModelInference:
    # Initialize the inference engine
    co_occur = load_co_occur_inference_model()
    inference_engine = MultiModelInference([co_occur])
    return inference_engine
