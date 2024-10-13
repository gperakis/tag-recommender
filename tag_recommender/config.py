from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
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


class Tag2VecSettings(BaseSettings):
    vector_size: int = 100
    window: int = 3
    min_count: int = 5
    workers: int = 16
    sg: int = 1  # Skip-gram. If 0, then CBOW.
    epochs: int = 200
    sorted_vocab: int = 1


class CountMinSketchSettings(BaseSettings):
    depth: int = 5
    width: int = 10000


class ModelSettings(BaseSettings):
    input_file: str = Field(
        default="data/full_dataset.csv", description="Path to the input file."
    )
    normalize: bool = Field(
        default=True, description="Whether to normalize tags during preprocessing."
    )
    datasets_dir: Path | None = Field(
        default=Path("artifacts", "datasets"),
        description="Directory to save split datasets.",
    )
    save_dir: Path | None = Field(
        default=Path("artifacts", "models"),
        description="Directory to save split datasets.",
    )


class SplittingSettings(BaseSettings):
    train_size: float = Field(default=0.8, description="Training set size.")
    val_size: float = Field(default=0.10, description="Validation set size.")
    test_size: float = Field(default=0.10, description="Test set size.")
    random_state: int = Field(
        default=42, description="Random seed for reproducibility."
    )


app_settings = AppSettings()
tag2vec_settings = Tag2VecSettings()
model_settings = ModelSettings()
splitting_settings = SplittingSettings()
cms_settings = CountMinSketchSettings()
