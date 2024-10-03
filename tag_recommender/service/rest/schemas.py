from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Request model for recommending tags."""

    tags: str = Field(
        ...,
        description="A comma separated list of tags for which we need recommendations.",
        examples=["art,photography", "nature", "creative-writing,digital-art"],
    )
    num_tags: int | None = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of recommended tags to fetch. Must be at least 1.",
        examples=[3, 5, None],
    )


class Status(BaseModel):
    """Model representing the status of an API response."""

    code: int = Field(
        ..., description="HTTP status code of the response.", examples=[200, 400]
    )
    message: str = Field(
        ...,
        description="Status message describing the result of the operation.",
        examples=["Recommendation successful."],
    )


class RecommendedItem(BaseModel):
    """Model representing a recommended item."""

    tag: str = Field(
        ...,
        description="Recommended tag.",
        examples=["digital-art", "creative-writing", "nature"],
    )
    score: float = Field(
        ...,
        description="Score of the recommendation.",
        examples=[0.9, 0.8, 0.7],
        ge=0.0,
        le=1.0,
    )
    # input_tags: List[str] = Field(
    #     ...,
    #     description="List of input tags used for the recommendation.",
    #     examples=[["art", "photography"], ["nature"],
    #     ["creative-writing", "digital-art"]],
    # )


class RecommendResponse(BaseModel):
    """Response model for recommending tags."""

    tags: list[RecommendedItem] | None = Field(
        ...,
        description="List of recommended tags based on input.",
        examples=["digital-art", "creative-writing", "nature"],
    )
    status: Status = Field(
        ...,
        description="Status information of the response.",
        examples=[{"code": 200, "message": "Recommendation successful."}],
    )
