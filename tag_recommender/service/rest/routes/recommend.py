import logging

from fastapi import APIRouter, HTTPException

from tag_recommender.recommend.factories import create_inference_engine
from tag_recommender.service.rest.schemas import (
    RecommendRequest,
    RecommendResponse,
    Status,
)
from tag_recommender.utils.text import split_tags

logger = logging.getLogger(__name__)

recommender_router = APIRouter(tags=["Recommender"])

# Initialize the inference engine
inference_engine = create_inference_engine()


def recommend_tags(tags: str, topn: int = 5):
    """
    Recommend tags based on input tags.

    Parameters
    ----------
    tags : str
        The input tags for which recommendations are generated.
    topn : int
        The number of top recommendations to return.

    Returns
    -------
    list[str]
    """
    logger.info(f"Recommendation request received for tags: {tags}")

    tags_array = split_tags(tags)
    recommended_tags = inference_engine.predict(tags=tags_array, topn=topn)
    recommended_tags = [
        {"tag": tag, "score": score}
        for tag, score in recommended_tags
        if tag not in tags_array
    ]

    response = RecommendResponse(
        input_tags=tags,
        tags=recommended_tags,
        status=Status(code=200, message="Recommendation successful."),
    )

    return response


router = APIRouter()


@recommender_router.post("/recommend", response_model=RecommendResponse)
async def recommend_tags_endpoint(request: RecommendRequest):
    try:
        return recommend_tags(request.tags, request.num_tags)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
