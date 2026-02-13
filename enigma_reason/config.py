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

    # Phase 4: Reasoning confidence weights
    confidence_weight_evidence: float = 0.25
    confidence_weight_rate: float = 0.15
    confidence_weight_diversity: float = 0.20
    confidence_weight_anomaly: float = 0.30
    confidence_weight_burst: float = 0.10
    confidence_evidence_saturation: float = 10.0
    confidence_rate_saturation: float = 10.0
    confidence_diversity_saturation: float = 3.0

    # Phase 4: Trend detection
    trend_rate_rise_factor: float = 1.5
    trend_rate_fall_factor: float = 2.0
    trend_recent_count: int = 3

    model_config = {"env_prefix": "ENIGMA_"}


settings = Settings()
