from abc import ABC, abstractmethod
from typing import Any

from tag_recommender.config import ModelConfig
from tag_recommender.process.split import DataSplitter


class BaseMLModel(ABC):
    def __init__(
        self,
        settings: ModelConfig,
        splitter: DataSplitter | None = None,
        evaluator: Any | None = None,
    ):
        """
        Initialize the BaseMLModel.

        Parameters
        ----------
        settings : ModelConfig
            The settings to use for the model.
            - input_file: str
                The path to the input file.
            - normalize: bool
                Whether to normalize the tags or not.
            - save_dir: str
                The directory to save the model and other artifacts.
            - random_state: int
                The random state to use for splitting the dataset

        splitter : DataSplitter | None (default: None)
            The DataSplitter object to use for splitting the dataset.
            The splitter can be None in the inference mode.

        evaluator : Any | None (default: None)
            The evaluator object to use for evaluating the model.
            The evaluator can be None in the inference mode.
        """
        self.input_file = settings.input_file
        self.normalize = settings.normalize
        self.save_dir = settings.save_dir

        self.data_splitter = splitter
        self.evaluator = evaluator

        self.model = None

    def preprocess(self) -> None:
        """
        This method preprocesses the dataset.
        It splits the dataset into train, validation, and test sets.
        It also normalizes the tags if the `normalize` parameter is set to True.
        The datasets are saved if the `save_dir` parameter is provided.

        Returns
        -------
        None
        """
        if self.data_splitter is None:
            raise ValueError("DataSplitter object is not provided.")

        self.data_splitter.preprocess(
            input_file=self.input_file,
            normalize=self.normalize,
            save_dir=self.save_dir,
        )

    def create_corpus(self):
        if self.data_splitter.df_train is None:
            raise ValueError(
                "The dataset has not been preprocessed. "
                "Please preprocess the dataset first."
            )
        self.data_splitter.create_corpus()

    @property
    def train_corpus(self):
        return self.data_splitter.train_corpus

    @property
    def validation_corpus(self):
        return self.data_splitter.validation_corpus

    @property
    def test_corpus(self):
        return self.data_splitter.test_corpus

    @abstractmethod
    def train(self):
        """Train the model using the dataset."""
        pass

    @abstractmethod
    def evaluate(self, corpus: list[str] | list[list[str]] | None):
        """
        Evaluate the model on the dataset.

        Parameters
        ----------
        corpus : list[str] | list[list[str]] | None
            The input corpus where each entry is a comma-separated list of hashtags or a
            list of hashtags.
            If the input is a list of lists, the hashtags are already normalized.
            If the input is a list of strings, the hashtags will be split
            and normalized.

        Returns
        -------
        """
        pass

    @abstractmethod
    def save_model(self):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model is trained yet!")
        # save the model

    @abstractmethod
    def load_model(self):
        """Load a saved model from a file."""
        pass

    @abstractmethod
    def recommend(self, tag: str, topn: int = 3) -> list[tuple[str, float]]:
        """
        Recommend similar tags for a given tag.

        Parameters
        ----------
        tag : str
            The tag for which to recommend similar tags.
        topn : int (default: 3)
            The number of similar tags to recommend.

        Returns
        -------
        list[tuple[str, float]]
            The list of similar tags with their similarity scores.
            E.g. [(tag1, score1), (tag2, score2), ...]
        """
        pass

    @abstractmethod
    def tag_exists(self, tag: str) -> bool:
        """
        Check if a tag exists in the model.

        Parameters
        ----------
        tag : str

        Returns
        -------
        bool
            True if the tag exists in the model, False otherwise.
        """
        pass

    @abstractmethod
    def recommend_many(
        self, tags: list[str], topn: int = 3
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Recommend similar tags for a list of tags.

        Parameters
        ----------
        tags : list[str]
            The list of tags for which to recommend similar tags.
        topn : int (default: 3)
            The number of similar tags to recommend.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            A dictionary where the key is the input tag and the value is a list of
            similar tags with their similarity scores.
            E.g. {tag1: [(tag1_1, score1_1), (tag1_2, score1_2), ...],
                  tag2: [(tag2_1, score2_1), (tag2_2, score2_2), ...],
        """
        pass
