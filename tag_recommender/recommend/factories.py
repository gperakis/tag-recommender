from tag_recommender.config import cms_settings, model_settings
from tag_recommender.process.factories import create_data_splitter
from tag_recommender.recommend.co_occur.model import CoOccurrenceModel
from tag_recommender.recommend.co_occur.sketch import CountMinSketch
from tag_recommender.utils.evaluate import Evaluator


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
    evaluator = Evaluator(model_settings.normalize)
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
    return model
