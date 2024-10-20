import gzip
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, udf
from pyspark.sql.types import ArrayType, Row, StringType

from tag_recommender.config import ModelConfig, SparkConfig, TagRulesConfig
from tag_recommender.process.split import DataSplitter
from tag_recommender.recommend.association_rules.utils import create_rules_knn_dict
from tag_recommender.recommend.base import BaseMLModel
from tag_recommender.utils.text import to_snake_case_boosted

logger = logging.getLogger(__name__)


@udf(ArrayType(StringType()))
def remove_duplicates(arr):
    return sorted(set(arr))


class TagRules(BaseMLModel):
    def __init__(
        self,
        settings: ModelConfig,
        splitter: DataSplitter | None = None,
        evaluator: Any | None = None,
        config: SparkConfig | None = None,
        tag_rules_config: TagRulesConfig | None = None,
    ):
        """
        Initialize the Co-occurrence model.

        Parameters
        ----------
        settings : ModelConfig

        splitter : DataSplitter (default: None)
            The DataSplitter object to use for splitting the dataset.
            The splitter can be None in the inference mode.

        evaluator : Any (default: None)
            The evaluator object to use for evaluating the model.
            The evaluator can be None in the inference mode.
        """
        super().__init__(settings, splitter, evaluator)

        self.knn = {}
        self.spark_config = config or SparkConfig()
        self.tag_rules_config = tag_rules_config or TagRulesConfig()
        self.model = None
        self._session = None
        self.rules = None

    @property
    def session(self) -> SparkSession:
        """Get the Spark session lazily"""
        if self._session is None:
            logger.info(
                f"Creating a new Spark session with the "
                f"following config: {self.spark_config}"
            )

            self._session = (
                SparkSession.builder.master("local[*]")
                .appName("FrequentPatternsSpark")
                .config(
                    "spark.executor.memory", self.spark_config.spark_executor_memory
                )
                .config("spark.driver.memory", self.spark_config.spark_driver_memory)
                .config(
                    "spark.executor.memoryOverhead",
                    self.spark_config.spark_executor_memory_overhead,
                )
                .config(
                    "spark.sql.shuffle.partitions",
                    self.spark_config.spark_sql_shuffle_partitions,
                )
                .config(
                    "spark.driver.maxResultSize",
                    self.spark_config.spark_driver_max_result_size,
                )
                .getOrCreate()
            )

        return self._session

    def stop_session(self):
        """Stop the Spark session."""
        if self._session:
            self._session.stop()
            self._session = None

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
            "association_rules_s{}_c{}_l{}.{}".format(
                self.tag_rules_config.support,
                self.tag_rules_config.min_confidence,
                self.tag_rules_config.lift,
                extension,
            ),
        )
        return model_path

    @property
    def rules_path(self) -> Path:
        return self.base_path("csv")

    @property
    def knn_path(self) -> Path:
        return self.base_path("pkl.gz")

    def save_model(self):
        """Save the trained artifacts to disk."""
        if self.model is None:
            raise ValueError("No model found. Please train a model first.")

        logger.info(f"Saving rules to path: {self.rules_path}")
        self.rules.to_csv(self.rules_path, index=False)
        logger.info("Rules saved successfully")

        if not self.knn:
            raise ValueError(
                "No k-nearest neighbors found. Please train or load a model first."
            )

        logger.info(f"Saving top recommendations to: {self.knn_path}")
        with gzip.open(self.knn_path, "wb") as f:
            pickle.dump(self.knn, f)
        logger.info("K-top recommendations saved successfully")

    def load_model(self):
        """Load the trained artifacts from disk."""
        logger.info(f"Loading top recommendations from: {self.knn_path}")
        with gzip.open(self.knn_path, "rb") as f:
            self.knn = pickle.load(f)
        logger.info("K-top recommendations loaded successfully")

    def precalculate_knn(self, k=60) -> dict[tuple, tuple]:
        """
        Pre-calculate the k-nearest neighbors for each tag in the vocabulary.

        Due to the nature of the association rules, the k-nearest neighbors are
        pre-calculated and stored in a dictionary for faster retrieval.
        The keys are tuples of tags and the values are lists of tuples of the
        nearest neighbors and their confidence scores.

        Parameters
        ----------
        k : int (default: 60)
            The number of nearest neighbors to pre-calculate.

        Returns
        -------
        dict[tuple, tuple]
            The dictionary of k-nearest neighbors for each tag in the vocabulary
        """
        if self.model is None:
            raise ValueError("No model found. Please train or load a model first.")

        logger.info(
            "Pre-calculating k-nearest neighbors for each tag in the vocabulary."
        )
        self.knn = create_rules_knn_dict(self.rules, topn=k)

        return self.knn

    def train(self):
        self.preprocess()
        self.create_corpus()

        logger.info("Converting Training corpus to spark dataframe...")
        # convert the training dataset into a spark dataframe
        train_corpus_rdd = self.session.sparkContext.parallelize(self.train_corpus)

        # Create the DataFrame and remove empty arrays and
        # duplicates before repartitioning
        df_unified = train_corpus_rdd.map(lambda x: Row(tag_arrays=x)).toDF()
        df_unified = df_unified.filter(size(col("tag_arrays")) > 0)
        df_unified = df_unified.withColumn(
            "tag_arrays", remove_duplicates(col("tag_arrays"))
        )

        # Cache and repartition
        df_unified.cache()
        df_unified = df_unified.repartition(
            self.spark_config.spark_sql_shuffle_partitions
        )

        # Calculate the length of df_unified
        n_baskets = df_unified.count()
        logger.info(f"Number of baskets found for training: {n_baskets}")

        # Persist after repartitioning and caching for long-term storage
        df_unified.persist(StorageLevel.MEMORY_AND_DISK)

        # Adjust min support
        min_support = self.tag_rules_config.support / n_baskets
        logger.info(
            f"Min support threshold: {self.tag_rules_config.support}. "
            f"As a percentage: {min_support}"
        )

        # Fit FPGrowth model
        fp_growth = FPGrowth(
            itemsCol="tag_arrays",
            minSupport=min_support,
            minConfidence=self.tag_rules_config.min_confidence,
        )
        self.model = fp_growth.fit(df_unified)

        # Generate association rules
        rules = self.model.associationRules.orderBy(col("confidence").desc())
        rules = rules.filter(rules.lift >= self.tag_rules_config.lift)

        # Add support count and antecedent/consequent size columns
        rules = rules.withColumn("support_count", rules.support * n_baskets)
        rules = rules.withColumn("antecedent_size", size(col("antecedent")))
        rules = rules.withColumn("consequent_size", size(col("consequent")))

        # Keep only rules with at most 4 antecedents.
        # This is to avoid very long rules that are not very useful.
        rules = rules.filter(rules.antecedent_size <= 4)

        # Sort antecedents and consequents
        rules = rules.withColumn(
            "antecedent", udf(sorted, ArrayType(StringType()))(col("antecedent"))
        ).withColumn(
            "consequent", udf(sorted, ArrayType(StringType()))(col("consequent"))
        )

        # Convert rules to pandas
        self.rules = rules.toPandas()

        # Precalculate KNN and save model
        self.precalculate_knn()
        self.save_model()
        self.stop_session()

    def recommend(self, tag: str, topn: int = 3):
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
            The list of similar hashtags with their estimated co-occurrence counts.
        """
        if not self.knn:
            raise ValueError("No model found. Please train or load a model first.")

        if self.normalize:
            # some tags may be the same after normalization
            tag = to_snake_case_boosted(tag)

        # convert to tuple to match the keys in the self.knn
        return self.knn.get((tag,), [])[:topn]

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
        # in the self.knn the keys are tuples of tags
        return (tag,) in self.knn

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
        if not self.knn:
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
            # convert to tuple to match the keys in the self.knn
            # TODO handle tha cases where we have multiple tags as a key
            tag_res = self.knn.get((tag,), [])[:topn]
            results.update({orig_tag: tag_res for orig_tag in original_tags})

        return results
