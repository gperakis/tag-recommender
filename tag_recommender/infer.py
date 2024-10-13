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
