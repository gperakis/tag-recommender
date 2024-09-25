from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Tag Recommender API"
    app_version: str = "1.0.0"
    allowed_origins: list[str] = ["*"]

    enable_api_docs: bool = Field(
        alias="TUMBLR_ENABLE_API_DOCS",
        description="Flag to enable or disable API documentation.",
        default=True,
    )

    class Config:
        env_file = ".env"


settings = Settings()
