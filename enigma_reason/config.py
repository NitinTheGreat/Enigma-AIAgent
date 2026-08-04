"""
Module: enigma_reason/config.py

Application configuration loaded from environment variables.

Every switch in the ABLATION section below is read by Level 9, which runs a
2^4 factorial across the four epistemic control mechanisms. When a switch is
disabled the mechanism must be absent from the reasoning path entirely, not
weakened, so that a disabled configuration is behaviourally identical to one
where the mechanism was never written.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

from enigma_reason.foundation.clock import ClockMode


class Settings(BaseSettings):
    """Environment-backed settings. All variables carry the ENIGMA_ prefix."""

    app_name: str = "enigma-reason"
    debug: bool = False
    log_level: str = "INFO"

    # ENIGMA_CORS_ORIGINS. The dashboard fetches /health from the browser, so
    # without this the footer renders dashes for data that exists. Documented in
    # .env.example since Phase 1 but never implemented, recorded as E8 in
    # paper/EVIDENCE.md.
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
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

    # Phase 5: LangGraph orchestration
    graph_max_iterations: int = 3
    graph_convergence_threshold: float = 0.8
    graph_convergence_persistence: int = 2

    # Phase 5: Gemini LLM
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 1024

    # ── Level 2: clock domain ────────────────────────────────────────────
    # ENIGMA_CLOCK_MODE. See foundation/clock.py and EVIDENCE.md section D1.
    # "conflated" reproduces the original defect and exists for the Level 8
    # experiment. It must never be the default.
    clock_mode: ClockMode = ClockMode.SEPARATED

    # ── Level 2: ablation switches ───────────────────────────────────────
    # U: the permanent non-prunable UNKNOWN hypothesis.
    unknown_hypothesis_enabled: bool = True
    # S: the deterministic structural sanity gate.
    sanity_gate_enabled: bool = True
    # A: negative evidence decaying confidence faster than positive evidence.
    asymmetric_decay_enabled: bool = True
    # P: the sustained dominance requirement before convergence is permitted.
    persistence_required: bool = True

    # Belief inertia cap. 0.15 matches the largest single adjustment the
    # evaluation node can apply in one pass, which is the 0.1 anomaly boost
    # plus the 0.05 trend boost. A cap below that would suppress a legitimate
    # single-pass update; a cap above it could never bind. Set to a very large
    # number to disable the mechanism for ablation.
    max_confidence_delta: float = 0.15

    model_config = {"env_prefix": "ENIGMA_"}


settings = Settings()
