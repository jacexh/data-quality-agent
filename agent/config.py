from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "robot-uploads"
    minio_use_ssl: bool = False

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_review_margin: float = 0.1

    clarity_threshold: float = 0.6
    continuity_threshold: float = 0.6
    minimum_duration_seconds: float = 1.0

    webhook_auth_token: str = ""
    model_dir: str = "/app/models"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
