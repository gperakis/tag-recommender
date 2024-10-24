import logging
from typing import Any

import pandas as pd
import pytrec_eval
from tqdm import tqdm

from tag_recommender.infer import MultiModelInference
from tag_recommender.recommend.base import BaseMLModel
from tag_recommender.utils.text import normalize_hashtags

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        split_tags_func: callable = None,
    ):
        """
        Initialize the evaluator with the corpus and the trained model.

        Parameters
        ----------
        split_tags_func : callable | None (default: None)
            A function to split and normalize tags.
            If None, the default function `normalize_hashtags` is used.

        """
        self.split_tags_func = split_tags_func
        if self.split_tags_func is None:
            self.split_tags_func = normalize_hashtags

        self.ground_truth = None
        self.ground_truth_without_oov = None

    # Create a dictionary for relevance and neighbors
    def prepare_eval_data(
        self, corpus: list[str] | list[list[str]], model: BaseMLModel
    ) -> dict[str, Any]:
        """
        Prepare the evaluation data for pytrec_eval.

        Parameters
        ----------
        corpus : list[str] | list[list[str]]
            The input corpus where each entry is a comma-separated list of hashtags or a
            list of hashtags.
            If the input is a list of lists, the hashtags are already normalized.
            If the input is a list of strings, the hashtags will be split and normalized

        model : BaseMLModel
            The trained model to use for recommendation.

        Returns
        -------
        dict[str, Any]
            The evaluation data containing the following keys:
            - ground_truth: The ground truth relevance judgments.
            - ground_truth_without_oov: The ground truth relevance judgments without
              OOV hashtags.

        """
        if self.ground_truth and self.ground_truth_without_oov:
            logger.info("Evaluation data already prepared.")
            return dict(
                ground_truth=self.ground_truth,
                ground_truth_without_oov=self.ground_truth_without_oov,
            )

        logger.info("Preparing evaluation data")
        ground_truth = {}
        ground_truth_without_oov = {}

        for irow, entry in tqdm(
            enumerate(corpus),
            desc="Preparing evaluation data",
            total=len(corpus),
        ):
            # Handle both strings and lists
            if isinstance(entry, str):
                # Split (and normalize) the tags
                row_tags = self.split_tags_func(entry)
            else:
                # The tags are already split and normalized
                row_tags = entry

            # Initialize the dictionaries for this row
            row_key = f"row_{irow}"
            ground_truth[row_key] = {}
            ground_truth_without_oov[row_key] = {}

            # Ground truth relevance (1 for ALL hashtags in the same sample)
            row_tags_hash = {tag: 1 for tag in row_tags}

            if len(row_tags_hash) < 2:
                # Cannot evaluate with less than 2 tags
                continue

            for tag in row_tags_hash:
                if not model.tag_exists(tag):
                    continue

                # remove the current tag from the ground truth
                gt_h = row_tags_hash.copy()
                gt_h.pop(tag)

                # Ground truth relevance without OOV hashtags
                gt_without_oov_h = {htag: 1 for htag in gt_h if model.tag_exists(htag)}

                if gt_h:  # if the ground truth is not empty
                    ground_truth[row_key][tag] = gt_h

                if gt_without_oov_h:  # maybe all the tags are OOV
                    ground_truth_without_oov[row_key][tag] = gt_without_oov_h

        logger.info("Evaluation data prepared.")
        self.ground_truth = ground_truth
        self.ground_truth_without_oov = ground_truth_without_oov

        return dict(
            ground_truth=ground_truth,
            ground_truth_without_oov=ground_truth_without_oov,
        )

    @staticmethod
    def get_metric_names(ks: list[int] | None) -> list[str]:
        """
        Get the list of metric names to calculate.

        Parameters
        ----------
        ks : list[int] | None
            The list of K values to calculate Recall, nDCG, and MAP at.
            If None, the default values are [3, 5].

        Returns
        -------
        list[str]
            The list of metric names to calculate.
        """
        if ks is None:
            ks = [3, 5]

        eval_metrics = {"map", "ndcg"}
        for k in ks:
            eval_metrics.add(f"recall.{k}")
            eval_metrics.add(f"ndcg_cut.{k}")
            eval_metrics.add(f"map_cut.{k}")
        return sorted(eval_metrics)

    @staticmethod
    def recommend(tag: str, model: BaseMLModel) -> dict[str, float]:
        """
        Recommend tags for a given tag.

        Parameters
        ----------
        tag : str
            The tag for which to recommend similar tags.
        model : BaseMLModel
            The trained model to use for recommendation.

        Returns
        -------
        dict[str, float]
            The recommended tags with their scores.
        """
        preds_tuples = model.recommend(tag, topn=30)
        return {tag: score for tag, score in preds_tuples}

    def calculate_metrics(
        self, y_true: dict, eval_metrics: list[str], model: BaseMLModel
    ) -> pd.DataFrame:
        """
        Calculate the evaluation metrics using pytrec_eval.

        Parameters
        ----------
        y_true : dict[str, dict[str, int]]
            The ground truth relevance judgments.

        eval_metrics : list[str]
            The list of evaluation metrics to calculate.

        model : BaseMLModel
            The trained model to use for recommendation.

        Returns
        -------
        pd.DataFrame
            The aggregated metrics for all rows.
        """

        gt_agg_metrics = []
        for row, qrel in tqdm(y_true.items(), desc="Calculating metrics"):
            run = {key: self.recommend(key, model) for key in qrel}

            evaluator = pytrec_eval.RelevanceEvaluator(qrel, eval_metrics)
            row_metrics = evaluator.evaluate(run)
            gt_agg_metrics.extend(list(row_metrics.values()))

        gt_agg_metrics = pd.DataFrame(gt_agg_metrics)
        return gt_agg_metrics.describe().T.sort_index()

    def calculate_retrieval_metrics(
        self,
        model: BaseMLModel | MultiModelInference,
        corpus: list[str] | list[list[str]],
        ks: list[int] | None = None,
    ):
        """
        Calculate the evaluation metrics using pytrec_eval.

        Metrics include:
        - Mean Average Precision (MAP)
        - Normalized Discounted Cumulative Gain (nDCG)
        - Recall@K
        - nDCG@K
        - MAP@K

        Parameters
        ----------
        model : BaseMLModel
            The trained model to use for recommendation.

        corpus : list[str] | list[list[str]]
            The input corpus where each entry is a comma-separated list of hashtags or a
            list of hashtags.
            If the input is a list of lists, the hashtags are already normalized.
            If the input is a list of strings, the hashtags will be split and normalized

        ks : list[int] | None (default: None)
            The list of K values to calculate Recall, nDCG, and MAP at.
            If None, the default values are [3, 5].

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary containing the following keys:
            - gt_metrics: The aggregated metrics for all rows.
            - gt_metrics_without_oov: The aggregated metrics for all rows without
                OOV hashtags.
        """
        if corpus:
            logger.info("Preparing evaluation data using the provided corpus")
            self.prepare_eval_data(corpus=corpus, model=model)
        else:
            if self.ground_truth is None or self.ground_truth_without_oov is None:
                raise ValueError("Corpus is required for the evaluation run.")

        if ks is None:
            ks = [3, 5]

        eval_metrics = self.get_metric_names(ks)

        logger.info(f"Calculating metrics: {eval_metrics} for Ground Truth")
        gt_metrics = self.calculate_metrics(self.ground_truth, eval_metrics, model)

        print(gt_metrics)

        logger.info(f"Calculating metrics: {eval_metrics} for Ground Truth without OOV")
        gt_metrics_without_oov = self.calculate_metrics(
            self.ground_truth_without_oov, eval_metrics, model
        )

        print(gt_metrics_without_oov)

        return dict(
            gt_metrics=gt_metrics, gt_metrics_without_oov=gt_metrics_without_oov
        )
