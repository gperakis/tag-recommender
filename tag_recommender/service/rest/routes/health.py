from fastapi import APIRouter

from tag_recommender import __version__
from tag_recommender.service.rest.context import launch_date


health_router = APIRouter(prefix="/health")


@health_router.get("/", tags=["health"])
def health():
    return {
        "status": "Active",
        "deployment_date": launch_date.get().isoformat(),
        "release": __version__,
    }
