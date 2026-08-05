"""
Module: enigma_reason/replay/__init__.py

Offline execution of the reasoning stack with no network service.

Levels 8 and 9 need to run the same reasoning path the live server runs, many
hundreds of times, without FastAPI, without websockets and without waiting on
a replayer that emits signals in real time. This package provides that path.
"""

from enigma_reason.replay.offline import (
    MockLLM,
    OfflineReplay,
    ReplayResult,
    load_signals,
    mock_llm_factory,
)

__all__ = [
    "MockLLM",
    "OfflineReplay",
    "ReplayResult",
    "load_signals",
    "mock_llm_factory",
]
