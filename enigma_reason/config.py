"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "enigma-reason"
    debug: bool = False
    log_level: str = "INFO"
    situation_ttl_minutes: int = 30
    situation_dormancy_minutes: int = 10

    model_config = {"env_prefix": "ENIGMA_"}


settings = Settings()
