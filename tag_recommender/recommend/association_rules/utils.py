import logging
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

pd.set_option("display.max_columns", None)

logger = logging.getLogger(__name__)


def create_rules_knn_dict(rules: pd.DataFrame, topn: int = 60) -> dict:
    """
    Create a dictionary of KNN rules from the association rules DataFrame.

    Parameters
    ----------
    rules : pd.DataFrame
        A DataFrame containing the association rules.
    topn : int, default 60
        The number of top rules to store for each antecedent.

    Returns
    -------
    dict[tuple[str], list[tuple[str, float]]]
        A dictionary with the antecedents as the key and a list of top consequents
        with their confidence as the value.
    """
    # Group by antecedent and create a hash for each group
    grouped_rules = defaultdict(dict)

    for _, row in tqdm(
        rules.iterrows(), total=rules.shape[0], desc="Creating KNN dict"
    ):
        antecedent = tuple(row["antecedent"])
        consequent = row["consequent"][0]
        grouped_rules[antecedent][consequent] = (row["confidence"], row["lift"])

    # Sort the grouped results by confidence (descending) and lift (descending) in
    # case of a tie
    sorted_grouped_rules = {}

    for antecedent, consequents in tqdm(grouped_rules.items(), desc="Sorting KNN dict"):
        # Sort the consequents by confidence first, and by lift in case of a tie
        sorted_consequents = sorted(
            consequents.items(),  # Get the items (consequent, (confidence, lift))
            key=lambda x: (x[1][0], x[1][1]),
            # Sort by confidence (x[1][0]) and then lift (x[1][1])
            reverse=True,  # Sort in descending order
        )[:topn]

        # Store the sorted consequents with only the consequent and confidence in the
        # final result
        sorted_grouped_rules[antecedent] = [
            (consequent, confidence)
            for consequent, (confidence, _) in sorted_consequents
        ]
    logger.info(f"Created KNN dict with {len(sorted_grouped_rules)} antecedents.")
    return sorted_grouped_rules
