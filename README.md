# enigma-reason

Agentic reasoning layer for the **Enigma** distributed security system.

## Architecture

```
ML Service ──ws──▶ /ws/signals ──▶ LangGraph ──▶ /ws/decisions ──▶ Frontend
```

## Quick Start

```bash
pip install -e ".[dev]"
uvicorn enigma_reason.main:app --reload
```

## Run Tests

```bash
pytest
```
