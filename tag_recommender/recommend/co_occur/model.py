import itertools
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tag_recommender.config import ModelSettings
from tag_recommender.process.split import DataSplitter
from tag_recommender.recommend.base import BaseMLModel
from tag_recommender.recommend.co_occur.sketch import CountMinSketch
from tag_recommender.utils.text import to_snake_case_boosted

logger = logging.getLogger(__name__)


class CoOccurrenceModel(BaseMLModel):
    def __init__(
        self,
        settings: ModelSettings,
        splitter: DataSplitter | None = None,
        evaluator: Any | None = None,
        cms: CountMinSketch | None = None,
    ):
        """
        Initialize the Co-occurrence model.

        Parameters
        ----------
        settings : ModelSettings

        splitter : DataSplitter (default: None)
            The DataSplitter object to use for splitting the dataset.
            The splitter can be None in the inference mode.

        evaluator : Any (default: None)
            The evaluator object to use for evaluating the model.
            The evaluator can be None in the inference mode.

        cms : CountMinSketch (default: None)
            The CountMinSketch object to use for estimating co-occurrence counts.
            The CMS can be None in the inference mode.
        """
        super().__init__(settings, splitter, evaluator)
        self.cms = cms or CountMinSketch()
        self.hashtag_pairs = defaultdict(set)
        self.model = None
        self.knn = {}

        self.train_corpus = None
        self.validation_corpus = None
        self.test_corpus = None

    @property
    def base_path(self) -> Path:
        return Path(
            self.save_dir, f"co_occur_w{self.cms.width}_d{self.cms.depth}_model.pkl"
        )

    def extra_process(self):
        """
        Extra processing for the Co-occurrence model.

        Returns
        -------
        None
        """
        if self.df_train is None:
            raise ValueError(
                "The dataset has not been preprocessed. "
                "Please preprocess the dataset first."
            )

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

    def train(self):
        """
        Populate the Count-Min Sketch and hashtag pairs with co-occurring hashtags.
        """
        self.preprocess()
        self.extra_process()

        for hashtags in tqdm(
            self.train_corpus, desc="Populating CMS and hashtag pairs"
        ):
            for h1, h2 in itertools.combinations(hashtags, 2):
                # Update CMS for both combinations
                pair1 = f"{h1}_{h2}".encode()
                pair2 = f"{h2}_{h1}".encode()

                self.cms.update(pair1)
                self.cms.update(pair2)

                # Update hashtag pairs
                self.hashtag_pairs[h1].add(h2)
                self.hashtag_pairs[h2].add(h1)

        self.precalculate_knn()
        self.save_model()

    def __recommend(self, hashtag: str, topn: int = 3) -> list[tuple[str, int]]:
        """
        Recommend hashtags based on co-occurrence counts with the given hashtag.

        Parameters
        ----------
        hashtag : str
            The input hashtag to recommend similar hashtags for.
        topn : int (default: 3)
            Number of recommendations to return.

        Returns
        -------
        list of tuples
            A list of top N recommended hashtags with estimated co-occurrence counts.
        """
        if self.cms is None:
            raise ValueError("No model found. Please train or load a model first.")

        if hashtag not in self.hashtag_pairs:
            return []

        co_occurrence_counts = []
        for co_hashtag in self.hashtag_pairs[hashtag]:
            pair = f"{hashtag}_{co_hashtag}"
            estimated_count = self.cms.estimate(pair.encode("utf-8"))
            co_occurrence_counts.append((co_hashtag, estimated_count))

        # Sort by estimated co-occurrence count
        co_occurrence_counts.sort(key=lambda x: x[1], reverse=True)

        return co_occurrence_counts[:topn]

    def recommend(self, tag: str, topn: int = 3) -> list[tuple[str, float]]:
        """
        Recommend similar hashtags for a given hashtag.

        Parameters
        ----------
        tag : str
            The tag for which to recommend similar tags.
        topn : int (default: 3)
            The number of similar hashtags to recommend.

        Returns
        -------
        list[tuple[str, float]]
            The list of similar hashtags with their estimated co-occurrence counts.
        """
        if not self.knn:
            raise ValueError("No model found. Please train or load a model first.")

        if self.normalize:
            # some tags may be the same after normalization
            tag = to_snake_case_boosted(tag)

        return self.knn.get(tag, [])[:topn]

    def save_model(self):
        """Save the trained model to a file."""
        with open(self.base_path, "wb") as f:
            # Save only what is necessary for inference (knn)
            pickle.dump(self.knn, f)
            logger.info(f"Inference model saved to {self.base_path}")

    def load_model(self):
        """
        Load a saved model from a file.

        Returns
        -------
        None
        """
        with open(self.base_path, "rb") as f:
            logger.info(f"Loading inference model from {self.base_path}")
            self.knn = pickle.load(f)
            logger.info(f"Inference model loaded from {self.base_path}")

    def tag_exists(self, tag: str) -> bool:
        """
        Check if a tag exists in the model.
        Used in the model evaluation.

        Parameters
        ----------
        tag : str

        Returns
        -------
        bool
            True if the tag exists in the model, False otherwise.
        """
        return tag in self.knn

    def evaluate(self, corpus: list[str] | list[list[str]] | None = None):
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
        if not self.evaluator:
            raise ValueError("No evaluator found. Please provide an evaluator.")

        if not self.knn:
            raise ValueError("No model found. Please train or load a model first.")

        if not corpus and not self.test_corpus:
            raise ValueError("No corpus provided for evaluation.")

        if not corpus:
            corpus = self.validation_corpus

        # Prepare data for pytrec_eval
        eval_data = self.evaluator.calculate_retrieval_metrics(
            self, corpus=corpus, ks=[3, 5]
        )
        return eval_data

    def precalculate_knn(self, k=30):
        """
        Pre-calculate the k-nearest neighbors for all hashtags in the vocabulary.
        This is useful for faster inference.

        Parameters
        ----------
        k : int (default: 30)
            The number of nearest neighbors to calculate.

        Returns
        -------
        None
        """
        if not self.hashtag_pairs:
            raise ValueError("No model found. Please train or load a model first.")

        logger.info(
            "Pre-calculating k-nearest neighbors for each tag in the vocabulary."
        )

        for hashtag in tqdm(self.hashtag_pairs, desc="Calculating KNN"):
            self.knn[hashtag] = self.__recommend(hashtag, topn=k)

    def recommend_many(
        self, tags: list[str], topn: int = 3
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Recommend similar hashtags for a list of hashtags.

        Parameters
        ----------
        tags : list[str]
            The list of tags for which to recommend similar tags.
        topn : int (default: 3)
            The number of similar hashtags to recommend.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            A dictionary where each key is a tag
            and the value is a list of similar hashtags
            with their estimated co-occurrence counts.
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
