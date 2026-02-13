"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "enigma-reason"
    debug: bool = False
    log_level: str = "INFO"

    # WebSocket origins allowed to connect
    cors_origins: list[str] = ["*"]

    model_config = {"env_prefix": "ENIGMA_"}


settings = Settings()
