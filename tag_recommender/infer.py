from collections import defaultdict

from tag_recommender.recommend.base import BaseMLModel


class MultiModelInference:
    def __init__(self, models: list[BaseMLModel], k=60):
        """
        Initialize the inference class with a list of models

        Parameters
        ----------
        models : list[BaseMLModel]
            A list of models that inherit from BaseMLModel and implement the
            `recommend_many` method.

        k : int (default: 60)
            The RRF constant to use.
        """
        self.models = models
        self.k = k

    def tag_exists(self, tag):
        """
        Helper method for the evaluation of the combined methodology

        Returns
        -------
        bool
            True if the tag exists in any of the models, False otherwise
        """
        exists = False
        for model in self.models:
            if model.tag_exists(tag):
                exists = True
                break

        return exists

    def recommend(self, tag, topn=30):
        """
        Recommend tags for a given tag using Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        tag : str
            The tag for which to recommend similar tags.
        topn : int (default: 30)
            The number of similar hashtags to recommend.

        Returns
        -------

        """
        rrf_scores = defaultdict(float)

        for model in self.models:
            recommendations = model.recommend(tag, topn)
            for rank, (recommended_tag, _) in enumerate(recommendations, start=1):
                rrf_scores[recommended_tag] += 1 / (self.k + rank)

        sorted_recommendations = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_recommendations[:topn]

    def predict(self, tags: list[str], topn=10):
        """
        Infer tags using Reciprocal Rank Fusion (RRF) for multiple input tags.

        Parameters
        ----------
        tags : list[str]
            The input tags for which recommendations are generated.

        topn : int (default: 10)
            The number of top recommendations to return from each model.

        Returns
        -------
        list
            A list of tuples where each tuple is (tag, final_rrf_score).
        """
        rrf_scores = defaultdict(float)

        # Loop through each model to get recommendations
        for model in self.models:
            # Get recommendations as a dict, with topn applied to each model
            recommendations: dict = model.recommend_many(tags, topn)

            # Iterate over the recommendations and apply RRF
            for key, recs in recommendations.items():
                # `key` can be a string (single tag) or a tuple (multiple tags)
                for rank, (recommended_tag, _) in enumerate(recs, start=1):
                    rrf_scores[recommended_tag] += 1 / (self.k + rank)

        # Sort the tags by the aggregated RRF score
        sorted_recommendations = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return the top N recommendations
        return sorted_recommendations[:topn]
