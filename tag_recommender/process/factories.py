from tag_recommender.config import splitting_settings
from tag_recommender.process.split import DataSplitter


def create_data_splitter() -> DataSplitter:
    """
    Instantiate a DataSplitter.

    Returns
    -------
    DataSplitter
    """
    return DataSplitter(**splitting_settings.dict())
