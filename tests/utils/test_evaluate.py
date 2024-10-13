from unittest.mock import MagicMock

import pytest

from tag_recommender.recommend.base import BaseMLModel
from tag_recommender.utils.evaluate import Evaluator
from tag_recommender.utils.text import normalize_hashtags


@pytest.fixture
def mock_model():
    # Create a mock model with tag_exists method
    model = MagicMock(spec=BaseMLModel)
    model.tag_exists.side_effect = lambda tag: tag in {"tag1", "tag2", "tag3"}
    return model


@pytest.fixture()
def evaluator():
    return Evaluator(split_tags_func=normalize_hashtags)


def test_prepare_eval_data_with_string_input(mock_model, evaluator):
    corpus = ["tag1, tag2, tag3"]
    result = evaluator.prepare_eval_data(corpus, mock_model)

    expected_ground_truth = {
        "row_0": {
            "tag1": {"tag2": 1, "tag3": 1},
            "tag2": {"tag1": 1, "tag3": 1},
            "tag3": {"tag1": 1, "tag2": 1},
        }
    }
    expected_ground_truth_without_oov = {
        # All the tags are in-vocabulary
        "row_0": expected_ground_truth["row_0"]
    }

    assert result["ground_truth"] == expected_ground_truth
    assert result["ground_truth_without_oov"] == expected_ground_truth_without_oov


def test_prepare_eval_data_with_list_input(mock_model, evaluator):
    corpus = [["tag1", "tag2", "tag3"]]
    result = evaluator.prepare_eval_data(corpus, mock_model)

    expected_ground_truth = {
        "row_0": {
            "tag1": {"tag2": 1, "tag3": 1},
            "tag2": {"tag1": 1, "tag3": 1},
            "tag3": {"tag1": 1, "tag2": 1},
        }
    }
    expected_ground_truth_without_oov = {
        # Again, all the tags are in-vocabulary
        "row_0": expected_ground_truth["row_0"]
    }

    assert result["ground_truth"] == expected_ground_truth
    assert result["ground_truth_without_oov"] == expected_ground_truth_without_oov


def test_prepare_eval_data_with_oov_tags(mock_model, evaluator):
    corpus = [["tag1", "tag2", "tag4"]]  # 'tag4' is OOV
    result = evaluator.prepare_eval_data(corpus, mock_model)

    expected_ground_truth = {
        "row_0": {
            "tag1": {"tag2": 1, "tag4": 1},
            "tag2": {"tag1": 1, "tag4": 1},
            # 'tag4' is OOV in the model; it will be skipped
        }
    }
    expected_ground_truth_without_oov = {
        "row_0": {
            "tag1": {"tag2": 1},
            "tag2": {"tag1": 1},
        }
    }

    assert result["ground_truth"] == expected_ground_truth
    assert result["ground_truth_without_oov"] == expected_ground_truth_without_oov
