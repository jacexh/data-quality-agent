import warnings
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "robot-uploads"
    minio_use_ssl: bool = False

    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_review_margin: float = 0.1

    clarity_threshold: float = 0.6
    continuity_threshold: float = 0.6
    minimum_duration_seconds: float = 1.0

    webhook_auth_token: str = ""
    model_dir: str = "/app/models"
    log_level: str = "INFO"

    max_queue_size: int = Field(default=100, gt=0)
    worker_count: int = Field(default=4, gt=0)
    frame_sample_rate: int = Field(default=30, gt=0)

    camera_topics: list[str] = []
    audio_topics: list[str] = []
    camera_pass_strategy: Literal["all", "any", "majority"] = "all"
    audio_pass_strategy: Literal["all", "any", "majority"] = "all"
    max_frames_per_topic: int = Field(default=300, gt=0)
    max_concurrent_topics: int = Field(default=4, gt=0)
    llm_max_concurrent_calls: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def warn_if_no_api_key(self) -> "Settings":
        if not self.anthropic_api_key:
            warnings.warn(
                "ANTHROPIC_API_KEY is not set — LLM judgment will be skipped and "
                "fall back to detector results.",
                stacklevel=2,
            )
        return self


settings = Settings()
