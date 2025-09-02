"""
Core configuration settings for the train traffic control system.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database settings
    database_url: str = "postgresql://user:password@localhost/train_control"
    database_url_async: Optional[str] = None

    # API settings
    api_v1_prefix: str = "/api"
    project_name: str = "Train Traffic Control System"
    version: str = "0.1.0"

    # Optimization settings
    optimization_timeout_seconds: int = 30
    max_trains_per_optimization: int = 100

    # WebSocket settings
    websocket_timeout: int = 60

    # Metrics settings
    metrics_retention_days: int = 30

    debug: bool = False
    log_level: str = "INFO"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    def __post_init__(self):
        if not self.database_url_async:
            # Convert PostgreSQL URL to async version
            self.database_url_async = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )


settings = Settings()
