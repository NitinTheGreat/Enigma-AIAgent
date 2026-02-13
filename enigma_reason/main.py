"""enigma-reason — Situation Memory, Temporal Awareness, Adapters & Reasoning.

This is the application entry point.  It wires the SituationStore,
AdapterRegistry, ReasoningEngine, and WebSocket endpoints together.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import FastAPI

from enigma_reason.adapters.auth import AuthAnomalyAdapter
from enigma_reason.adapters.network import NetworkAnomalyAdapter
from enigma_reason.adapters.registry import AdapterRegistry
from enigma_reason.adapters.video import VideoDetectionAdapter
from enigma_reason.api.analyze import create_analyze_router
from enigma_reason.api.ws_raw_signal import create_raw_signal_router
from enigma_reason.api.ws_signal import create_signal_router
from enigma_reason.config import settings
from enigma_reason.core.reasoning_engine import (
    ConfidenceWeights,
    ReasoningEngine,
    TrendConfig,
)
from enigma_reason.store.correlation import EntityCorrelation
from enigma_reason.store.situation_store import SituationStore

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── Reasoning Engine ─────────────────────────────────────────────────────────

reasoning_engine = ReasoningEngine(
    weights=ConfidenceWeights(
        evidence=settings.confidence_weight_evidence,
        rate=settings.confidence_weight_rate,
        diversity=settings.confidence_weight_diversity,
        anomaly=settings.confidence_weight_anomaly,
        burst=settings.confidence_weight_burst,
        evidence_saturation=settings.confidence_evidence_saturation,
        rate_saturation=settings.confidence_rate_saturation,
        diversity_saturation=settings.confidence_diversity_saturation,
    ),
    trend_config=TrendConfig(
        rate_rise_factor=settings.trend_rate_rise_factor,
        rate_fall_factor=settings.trend_rate_fall_factor,
        recent_count=settings.trend_recent_count,
    ),
    burst_factor=settings.burst_factor,
    burst_recent_count=settings.burst_recent_count,
    quiet_window=timedelta(minutes=settings.quiet_window_minutes),
)

# ── State ────────────────────────────────────────────────────────────────────

store = SituationStore(
    ttl=timedelta(minutes=settings.situation_ttl_minutes),
    dormancy_window=timedelta(minutes=settings.situation_dormancy_minutes),
    correlation=EntityCorrelation(),
    burst_factor=settings.burst_factor,
    burst_recent_count=settings.burst_recent_count,
    quiet_window=timedelta(minutes=settings.quiet_window_minutes),
    reasoning_engine=reasoning_engine,
)

# ── Adapter Registry ────────────────────────────────────────────────────────

registry = AdapterRegistry()
registry.register(NetworkAnomalyAdapter())
registry.register(AuthAnomalyAdapter())
registry.register(VideoDetectionAdapter())

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Situation Memory, Temporal Awareness, Adapters & Reasoning",
    version="0.4.0",
)

# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(create_signal_router(store))
app.include_router(create_raw_signal_router(store, registry))
app.include_router(create_analyze_router(
    store,
    reasoning_engine,
    burst_factor=settings.burst_factor,
    burst_recent_count=settings.burst_recent_count,
    quiet_window=timedelta(minutes=settings.quiet_window_minutes),
))


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    ts = await store.temporal_summary()
    rs = await store.reasoning_summary()
    return {
        "status": "ok",
        "phase": 6.1,
        "active_situations": ts.active_situations,
        "dormant_situations": ts.dormant_situations,
        "bursting_situations": ts.bursting_situations,
        "quiet_situations": ts.quiet_situations,
        "max_event_rate": ts.max_event_rate,
        "escalating_situations": rs.escalating_situations,
        "stable_situations": rs.stable_situations,
        "deescalating_situations": rs.deescalating_situations,
        "average_confidence": rs.average_confidence,
        "max_confidence": rs.max_confidence,
        "adapters": registry.stats,
        "total_adapted": registry.total_accepted,
        "total_rejected": registry.total_rejected,
    }
