from fastapi import APIRouter

base_router = APIRouter()


@base_router.get("/")
def welcome():
    msg = "Welcome to the Tag Recommender API. Visit /docs for API documentation."
    return msg
