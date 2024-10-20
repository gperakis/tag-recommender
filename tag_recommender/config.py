from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
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
        extra = "ignore"


class Tag2VecConfig(BaseSettings):
    vector_size: int = 100
    window: int = 3
    min_count: int = 5
    workers: int = 16
    sg: int = 1  # Skip-gram. If 0, then CBOW.
    epochs: int = 200
    sorted_vocab: int = 1

    class Config:
        env_prefix = "TAG2VEC_"
        env_file = ".env"
        extra = "ignore"


class CountMinSketchConfig(BaseSettings):
    depth: int = 5
    width: int = 10000


class SparkConfig(BaseSettings):
    spark_executor_memory: str = "8g"
    spark_driver_memory: str = "8g"
    spark_executor_memory_overhead: str = "2g"
    spark_sql_shuffle_partitions: int = 2000
    spark_driver_max_result_size: str = "4g"

    class Config:
        env_prefix = "SPARK_"
        env_file = ".env"
        extra = "ignore"


class TagRulesConfig(BaseSettings):
    support: int = 250
    min_confidence: float = 0.05
    lift: float = 1.0

    class Config:
        env_prefix = "TAG_RULES_"
        env_file = ".env"
        extra = "ignore"


class ModelConfig(BaseSettings):
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


app_config = AppConfig()
tag2vec_config = Tag2VecConfig()
model_config = ModelConfig()
splitting_config = SplittingSettings()
cms_config = CountMinSketchConfig()
spark_config = SparkConfig()
tag_rules_config = TagRulesConfig()
