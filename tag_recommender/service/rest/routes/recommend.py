import logging

from fastapi import APIRouter, HTTPException

from tag_recommender.service.rest.schemas import (
    RecommendRequest,
    RecommendResponse,
    Status,
)

logger = logging.getLogger(__name__)

recommender_router = APIRouter(tags=["Recommender"])


def recommend_tags(tags: str, num_tags: int = 5):
    """
    Recommend tags based on input tags.
    TODO Move this function outside

    Parameters
    ----------x
    tags : str
    num_tags : int

    Returns
    -------
    list[str]
    """
    logger.info(f"Recommendation request received for tags: {tags}")
    # Placeholder for actual recommendation

    data = [
        {"tag": "digital-art", "score": 0.9},
        {"tag": "creative-writing", "score": 0.8},
    ][:num_tags]

    return data


router = APIRouter()


@recommender_router.post("/recommend", response_model=RecommendResponse)
async def recommend_tags_endpoint(request: RecommendRequest):
    try:
        recommended_tags = recommend_tags(request.tags, request.num_tags)
        return RecommendResponse(
            tags=recommended_tags,
            status=Status(code=200, message="Recommendation successful."),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
