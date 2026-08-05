"""enigma-reason — Situation Memory, Temporal Awareness, Adapters & Reasoning.

This is the application entry point.  It wires the SituationStore,
AdapterRegistry, ReasoningEngine, and WebSocket endpoints together.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from enigma_reason.observability.backpressure import AnalysisPool
from enigma_reason.observability.latency import LatencyRecorder
from enigma_reason.observability.run_log import RunLogWriter
from enigma_reason.adapters.auth import AuthAnomalyAdapter
from enigma_reason.adapters.network import NetworkAnomalyAdapter
from enigma_reason.adapters.registry import AdapterRegistry
from enigma_reason.adapters.unsw_threat import UNSWThreatAdapter
from enigma_reason.adapters.video import VideoDetectionAdapter
from enigma_reason.api.analyze import create_analyze_router
from enigma_reason.api.ws_dashboard import DashboardManager, create_dashboard_router
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
    clock_mode=settings.clock_mode,
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
    clock_mode=settings.clock_mode,
)

# ── Adapter Registry ────────────────────────────────────────────────────────

registry = AdapterRegistry()
registry.register(UNSWThreatAdapter())
registry.register(NetworkAnomalyAdapter())
registry.register(AuthAnomalyAdapter())
registry.register(VideoDetectionAdapter())

# ── Instrumentation ──────────────────────────────────────────────────────────

latency = LatencyRecorder()
analysis_pool = AnalysisPool(
    max_concurrent=settings.analysis_max_concurrent,
    max_pending=settings.analysis_max_pending,
    latency_recorder=latency,
)
run_log = RunLogWriter(settings.run_log_path) if settings.run_log_path else None

# ── Dashboard Manager ────────────────────────────────────────────────────────

dashboard = DashboardManager(
    reasoning_engine=reasoning_engine,
    burst_factor=settings.burst_factor,
    burst_recent_count=settings.burst_recent_count,
    quiet_window=timedelta(minutes=settings.quiet_window_minutes),
    dormancy_window=timedelta(minutes=settings.situation_dormancy_minutes),
    ttl=timedelta(minutes=settings.situation_ttl_minutes),
    latency=latency,
    run_log=run_log,
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Situation Memory, Temporal Awareness, Adapters & Reasoning",
    version="0.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(create_signal_router(
    store,
    dashboard_manager=dashboard,
    adapter_registry=registry,
    analysis_pool=analysis_pool,
    latency=latency,
))
app.include_router(create_raw_signal_router(store, registry, dashboard_manager=dashboard))
app.include_router(create_dashboard_router(dashboard))
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


# ── Metrics ──────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics() -> dict:
    """Expose per stage latency and the analysis backlog.

    Section C1 of paper/EVIDENCE.md records that no endpoint reported timing,
    queue depth or throughput, so the only latency figure anywhere was a
    browser round trip to a counter endpoint. This is the replacement.

    Stage timings are reported separately rather than summed, because the
    language model call dominates the other five by one to two orders of
    magnitude and an aggregate would conceal that.
    """
    return {
        "latency_ms": latency.summary(),
        "analysis_pool": analysis_pool.snapshot().to_dict(),
        "run_log": (
            {"path": settings.run_log_path, "written": run_log.written,
             "dropped": run_log.dropped}
            if run_log is not None
            else {"enabled": False}
        ),
        "dashboard_clients": dashboard.client_count,
    }
