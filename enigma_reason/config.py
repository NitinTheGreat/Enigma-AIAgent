"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "enigma-reason"
    debug: bool = False
    log_level: str = "INFO"
    situation_ttl_minutes: int = 30
    situation_dormancy_minutes: int = 10

    # Phase 2: Temporal awareness
    burst_factor: float = 3.0
    burst_recent_count: int = 3
    quiet_window_minutes: int = 5

    model_config = {"env_prefix": "ENIGMA_"}


settings = Settings()
