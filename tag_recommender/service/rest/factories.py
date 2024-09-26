from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tag_recommender.config import settings
from tag_recommender.logger import initialize_logging
from tag_recommender.service.rest.context import launch_date


def create_fastapi_application() -> FastAPI:
    """
    Factory function to create a FastAPI application and register the project's routes

    Returns
    -------
    FastAPI
        the FastAPI application
    """
    path = Path("config", "logging.yaml")
    initialize_logging(path)

    docs_args = dict(
        title=settings.app_name,
        description="API for a recommender system that suggests tags based on input "
        "tags.",
        version=settings.app_version,
    )
    if not settings.enable_api_docs:
        # Disable API documentation
        docs_args = {"docs_url": None, "redoc_url": None}

    api = FastAPI(**docs_args)

    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routers(api)
    launch_date.set(datetime.now(tz=timezone.utc))
    return api


def register_routers(api: FastAPI) -> None:
    """
    Registers the project's routes to the FastAPI application


    Parameters
    ----------
    api : FastAPI
        the FastAPI application on which to register the routes

    Returns
    -------
    None
    """
    from tag_recommender.service.rest.routes.base import base_router
    from tag_recommender.service.rest.routes.health import health_router
    from tag_recommender.service.rest.routes.recommend import recommender_router

    api.include_router(base_router)
    api.include_router(health_router)
    api.include_router(recommender_router)


if __name__ == "__main__":
    create_fastapi_application()
