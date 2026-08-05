"""
Module: enigma_reason/observability/__init__.py

Server side instrumentation for the reasoning layer.

Before Level 6 there was none, which is recorded as section C1 of
paper/EVIDENCE.md and blocked every figure in the paper except the
architecture diagram. Every value the paper needs was already present in the
returned graph state; what was missing was a writer.

Nothing in this package is permitted to change reasoning behaviour or to fail
an analysis. Writers swallow their own exceptions and report a drop count
instead of propagating.
"""

from enigma_reason.observability.backpressure import AnalysisPool, PoolSnapshot
from enigma_reason.observability.latency import LatencyRecorder, Stage
from enigma_reason.observability.llm_cache import CachingLLMFactory, CacheStats
from enigma_reason.observability.manifest import build_run_manifest, environment_fingerprint
from enigma_reason.observability.run_log import IterationRecorder, RunLogWriter

__all__ = [
    "AnalysisPool",
    "PoolSnapshot",
    "LatencyRecorder",
    "Stage",
    "CachingLLMFactory",
    "CacheStats",
    "build_run_manifest",
    "environment_fingerprint",
    "IterationRecorder",
    "RunLogWriter",
]
