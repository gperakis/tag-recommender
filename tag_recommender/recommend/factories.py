from tag_recommender.config import cms_settings, model_settings
from tag_recommender.infer import MultiModelInference
from tag_recommender.process.factories import create_data_splitter
from tag_recommender.recommend.co_occur.model import CoOccurrenceModel
from tag_recommender.recommend.co_occur.sketch import CountMinSketch
from tag_recommender.utils.evaluate import Evaluator
from tag_recommender.utils.text import normalize_hashtags, split_tags


def create_csm() -> CountMinSketch:
    """Create a CountMinSketch instance."""
    return CountMinSketch(**cms_settings.dict())


def create_co_occur_model() -> CoOccurrenceModel:
    """
    Instantiate a CoOccurrenceModel

    Returns
    -------
    CoOccurrenceModel
    """
    splitter = create_data_splitter()
    split_fun = normalize_hashtags if model_settings.normalize else split_tags
    evaluator = Evaluator(split_fun)
    cms = create_csm()
    return CoOccurrenceModel(model_settings, splitter, evaluator, cms)


def train_co_occur_model():
    model = create_co_occur_model()
    model.train()
    model.evaluate()


def load_co_occur_inference_model() -> CoOccurrenceModel:
    """Load a Co-occurrence model for inference."""
    model = CoOccurrenceModel(model_settings)
    model.load_model()
    if not model.knn:
        raise ValueError("No model found. Please train or load a model first.")
    return model


def create_inference_engine() -> MultiModelInference:
    # Initialize the inference engine
    co_occur = load_co_occur_inference_model()
    inference_engine = MultiModelInference([co_occur])
    return inference_engine
