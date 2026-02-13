# enigma-reason

**Phase 1 — Situation Memory & Signal Grounding**

Receives structured anomaly signals over WebSocket, validates them, and organises
them into long-lived Situations (evidence containers with TTL-based expiry).

## Architecture

```
ML Service ──ws──▶ /ws/signal ──▶ SituationStore ──▶ Situation(evidence[])
```

## Quick Start

```bash
pip install -e ".[dev]"
uvicorn enigma_reason.main:app --reload
```

## Run Tests

```bash
pytest -v
```

## Project Structure

```
enigma_reason/
├── api/            # WebSocket transport endpoints
├── domain/         # Canonical models: Signal, Situation, enums
├── store/          # In-memory situation state management
├── foundation/     # Utilities: clock, ID generation
├── config.py       # Environment-based settings
└── main.py         # FastAPI entry point
```
