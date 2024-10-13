import pytest
from pydantic import ValidationError

from tag_recommender.service.rest.schemas import (
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
    Status,
)


def test_recommend_request_valid():
    data = {"tags": "art,photography", "num_tags": 3}
    req = RecommendRequest(**data)
    assert req.tags == "art,photography"
    assert req.num_tags == 3


def test_recommend_request_default_num_tags():
    data = {"tags": "nature"}
    req = RecommendRequest(**data)
    assert req.num_tags == 5  # default value


def test_recommend_request_invalid_num_tags_below_minimum():
    data = {"tags": "art", "num_tags": 0}
    with pytest.raises(ValidationError):
        RecommendRequest(**data)


def test_recommend_request_invalid_num_tags_over_maximum():
    # 50 is the maximum allowed value
    data = {"tags": "art", "num_tags": 51}
    with pytest.raises(ValidationError):
        RecommendRequest(**data)


def test_status_valid():
    data = {"code": 200, "message": "Recommendation successful."}
    status = Status(**data)
    assert status.code == 200
    assert status.message == "Recommendation successful."


def test_status_invalid_code():
    # invalid status code (non-integer)
    data = {"code": "invalid_code", "message": "Invalid status code."}
    with pytest.raises(ValidationError):
        Status(**data)


def test_recommended_item_valid():
    data = {"tag": "digital-art", "score": 0.9}
    item = RecommendedItem(**data)
    assert item.tag == "digital-art"
    assert item.score == 0.9


def test_recommended_item_invalid_score():
    # invalid score (greater than 1.0)
    data = {"tag": "creative-writing", "score": 1.5}
    with pytest.raises(ValidationError):
        RecommendedItem(**data)


def test_recommend_response_valid():
    data = {
        "input_tags": "art,photography",
        "tags": [
            {"tag": "digital-art", "score": 0.9},
            {"tag": "creative-writing", "score": 0.8},
        ],
        "status": {"code": 200, "message": "Recommendation successful."},
    }
    response = RecommendResponse(**data)
    assert response.status.code == 200
    assert response.status.message == "Recommendation successful."
    assert response.input_tags == "art,photography"
    assert response.tags[0].tag == "digital-art"
    assert response.tags[0].score == 0.9
    assert response.tags[1].tag == "creative-writing"
    assert response.tags[1].score == 0.8


def test_recommend_response_invalid_tags():
    # Test invalid tags (dict instead of a list)
    data = {
        "tags": {},
        "status": {"code": 200, "message": "Recommendation successful."},
    }
    with pytest.raises(ValidationError):
        RecommendResponse(**data)
