import gzip
import logging
import pickle
from collections import defaultdict
from pathlib import Path

from gensim.models import Word2Vec
from tqdm import tqdm

from tag_recommender.config import ModelConfig, tag2vec_config
from tag_recommender.process.split import DataSplitter
from tag_recommender.recommend.base import BaseMLModel
from tag_recommender.utils.text import to_snake_case_boosted

logger = logging.getLogger(__name__)


class Tag2VecModel(BaseMLModel):
    def __init__(self, settings: ModelConfig, splitter: DataSplitter, evaluator):
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

        splitter : DataSplitter
            The DataSplitter object to use for splitting the dataset.

        evaluator : Evaluator
            The Evaluator object to use for evaluating the model.
        """
        super().__init__(settings, splitter, evaluator)
        self.knn = {}
        self.train_corpus = None
        self.validation_corpus = None
        self.test_corpus = None

    def base_path(self, extension: str = "model") -> Path:
        """
        Get the base path for saving the model.

        Parameters
        ----------
        extension : str (default: 'model')
            The extension to use for the model file.

        Returns
        -------
        Path
        """
        # create the model path using the params
        model_path = Path(
            self.save_dir,
            "word2vec_vs{}_w{}_mc{}_sg{}_ep{}.{}".format(
                tag2vec_config.vector_size,
                tag2vec_config.window,
                tag2vec_config.min_count,
                tag2vec_config.sg,
                tag2vec_config.epochs,
                extension,
            ),
        )
        return model_path

    @property
    def model_path(self) -> Path:
        return self.base_path("model")

    @property
    def knn_path(self) -> Path:
        return self.base_path("pkl.gz")

    def save_model(self):
        """Save the trained artifacts to disk."""
        if self.model is None:
            raise ValueError("No model is trained yet!")

        logger.info(f"Saving model to path: {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_path))
        logger.info("Model saved successfully")

        if not self.knn:
            raise ValueError(
                "No k-nearest neighbors found. Please train or load a model first."
            )

        logger.info(f"Saving k-nearest neighbors to path: {self.knn_path}")
        with gzip.open(self.knn_path, "wb") as f:
            pickle.dump(self.knn, f)
        logger.info("k-nearest neighbors saved successfully")

    def load_model(self) -> Word2Vec:
        """
        Load a trained model from a file.

        Returns
        -------
        Word2Vec
            The trained Word2Vec model
        """
        if self.model:
            logger.info("Model already loaded. Returning the loaded model.")
            return self.model

        logger.info(f"Loading model from path: {self.model_path}")
        self.model = Word2Vec.load(str(self.model_path))
        logger.info("Model loaded successfully")

        logger.info("Loading k-nearest neighbors...")
        with gzip.open(self.knn_path, "rb") as f:
            self.knn = pickle.load(f)
        logger.info("k-nearest neighbors loaded successfully")
        return self.model

    def extra_process(self):
        """
        This method processes the dataset further.

        It prepares the corpus for training the Word2Vec model by concatenating the root
        tags and the re-blogged tags.

        Returns
        -------
        None
        """
        if self.df_train is None:
            raise ValueError(
                "The dataset has not been preprocessed. "
                "Please preprocess the dataset first."
            )

        rt = "root_tags"
        t = "tags"
        train_corpus = self.df_train[rt].tolist() + self.df_train[t].tolist()
        validation_corpus = self.df_val[rt].tolist() + self.df_val[t].tolist()
        test_corpus = self.df_test[rt].tolist() + self.df_test[rt].tolist()

        # get rid of empty lists
        self.train_corpus = [arr for arr in train_corpus if arr]
        self.validation_corpus = [arr for arr in validation_corpus if arr]
        self.test_corpus = [arr for arr in test_corpus if arr]

    def precalculate_knn(self, k=30):
        """
        Pre-calculate the k-nearest neighbors for each tag in the vocabulary.

        Parameters
        ----------
        k : int (default: 30)
            The number of nearest neighbors to pre-calculate.

        Returns
        -------

        """
        if not self.model:
            raise ValueError("No model found. Please train or load a model first.")

        logger.info(
            "Pre-calculating k-nearest neighbors for each tag in the vocabulary."
        )

        # TODO: We may try to use Annoy or Faiss for faster retrieval
        #  of nearest neighbors
        self.knn = {
            tag: self.model.wv.most_similar(tag, topn=k)
            for tag in tqdm(self.model.wv.key_to_index)
        }
        logger.info(
            f"Pre-calculation of k-nearest neighbors for {k} neighbors complete."
        )

        return self.knn

    def train(self) -> Word2Vec:
        """
        This method trains the Word2Vec model on the tags in the dataset.

        Returns
        -------
        Word2Vec
            The trained Word2Vec model.
        """
        self.preprocess()
        self.extra_process()
        logger.info("Training Word2Vec model...")
        self.model = Word2Vec(self.train_corpus, **tag2vec_config.model_dump())
        self.precalculate_knn()
        self.save_model()
        logger.info("Word2Vec model training complete.")

        return self.model

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
        """
        if self.model is None:
            raise ValueError("No model found. Please train or load a model first.")

        if self.normalize:
            # some tags may be the same after normalization
            tag = to_snake_case_boosted(tag)

        return self.knn.get(tag, [])[:topn]

    def tag_exists(self, tag: str) -> bool:
        """
        Check if a tag exists in the vocabulary.

        Parameters
        ----------
        tag : str
            The tag to check.

        Returns
        -------
        bool
            True if the tag exists in the vocabulary, False otherwise.
        """
        if not self.knn:
            raise ValueError(
                "No k-nearest neighbors found. Please train or load a model first."
            )
        return tag in self.knn

    def evaluate(self, corpus: list[str] | list[list[str]] | None = None):
        """
        Evaluate the model on the given corpus.

        Parameters
        ----------
        corpus : list[str] | list[list[str]] | None (default: None)
            The corpus to evaluate the model on. If None, the validation corpus is
            used.


        Returns
        -------

        """
        if not self.model:
            raise ValueError("No model found. Please train or load a model")

        if not corpus and not self.validation_corpus:
            raise ValueError("No corpus provided for evaluation")

        if not corpus:
            corpus = self.validation_corpus

        # Prepare data for pytrec_eval
        eval_data = self.evaluator.calculate_retrieval_metrics(
            self, corpus=corpus, ks=[3, 5]
        )
        return eval_data

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
            The dictionary of tags with their recommended similar tags and their
            similarity scores.
        """
        # we may have different tags that their normalized form is the same
        # we want to reduce the query time by normalizing the tags only once
        # and then mapping the normalized tags to the original tags
        norm2original = defaultdict(list)

        for tag in tags:
            if self.normalize:
                norm_tag = to_snake_case_boosted(tag)
                norm2original[norm_tag].append(tag)
            else:
                norm2original[tag].append(tag)

        results = {}
        for tag, original_tags in norm2original.items():
            tag_res = self.knn.get(tag, [])[:topn]
            results.update({orig_tag: tag_res for orig_tag in original_tags})

        return results
