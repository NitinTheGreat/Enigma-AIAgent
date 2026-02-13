# enigma-reason

**Situation Memory & Signal Grounding + Temporal Awareness**

Receives structured anomaly signals over WebSocket, validates them, organises
them into long-lived Situations, and exposes temporal observations about
evidence arrival patterns.

---

## Architecture

```
ML Service ──ws──▶ /ws/signal ──▶ SituationStore ──▶ Situation(evidence[], temporal metrics)
```

### Signal Flow

1. ML service connects to `/ws/signal` via WebSocket
2. Each JSON payload is validated into an immutable `Signal` model
3. The `SituationStore` groups signals using a pluggable `CorrelationStrategy`
4. Evidence is attached to the matched (or newly created) `Situation`
5. Temporal metrics update automatically on each attach
6. Ack is returned: `{ status, situation_id, evidence_count }`

### Situation Lifecycle

```
active ──(dormancy window)──▶ dormant ──(TTL)──▶ expired (removed)
   ▲                              │
   └──── new evidence ◀───────────┘
```

### Temporal Metrics (Phase 2)

Each situation exposes read-only temporal observations:

| Property | Description |
|---|---|
| `first_seen_at` | Timestamp of earliest evidence |
| `last_seen_at` | Timestamp of most recent evidence |
| `active_duration` | Seconds between first and last event |
| `event_intervals` | Time gaps between consecutive events |
| `event_rate` | Events per minute over active duration |
| `is_bursting()` | True if recent events arrive faster than historical average |
| `is_quiet()` | True if no events within configurable quiet window |
| `temporal_snapshot()` | Immutable `SituationTemporalSnapshot` for downstream consumption |

These are **observations, not conclusions** — no risk scores, no alerts.

---

## Endpoints

### `WebSocket /ws/signal`

Ingests structured anomaly signals from the ML detection service.

**Request** (JSON per message):

```json
{
  "signal_id": "uuid",
  "timestamp": "2026-02-13T14:00:00Z",
  "signal_type": "intrusion",
  "entity": { "kind": "user", "identifier": "alice" },
  "anomaly_score": 0.85,
  "confidence": 0.9,
  "features": ["login_burst", "geo_anomaly"],
  "source": "detector-alpha"
}
```

**Response** (JSON per message):

```json
{ "status": "accepted", "situation_id": "uuid", "evidence_count": 3 }
```

### `GET /health`

```json
{
  "status": "ok",
  "phase": 2,
  "active_situations": 5,
  "dormant_situations": 2,
  "bursting_situations": 1,
  "quiet_situations": 3,
  "max_event_rate": 12.5
}
```

---

## Project Structure

```
enigma_reason/
├── main.py                          # FastAPI entry point
├── config.py                        # Env-based settings (ENIGMA_ prefix)
├── api/
│   └── ws_signal.py                 # WebSocket transport
├── domain/
│   ├── enums.py                     # Signal types, entity kinds
│   ├── signal.py                    # Canonical immutable Signal model
│   ├── situation.py                 # Situation with lifecycle + temporal metrics
│   └── temporal.py                  # SituationTemporalSnapshot (frozen Pydantic)
├── store/
│   ├── correlation.py               # Pluggable correlation strategy
│   └── situation_store.py           # In-memory store + TemporalSummary
├── foundation/
│   ├── clock.py                     # Patchable UTC clock
│   └── identifiers.py              # UUID generation
├── adapters/
│   └── base.py                      # Abstract SignalAdapter (reserved)
tests/
├── test_signal.py                   # Signal validation tests
├── test_situation.py                # Situation lifecycle tests
├── test_store.py                    # Store ingestion + expiry tests
└── test_temporal.py                 # Temporal metrics, burst, quiet, snapshot tests
```

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ENIGMA_APP_NAME` | `enigma-reason` | Application name |
| `ENIGMA_DEBUG` | `false` | Debug mode |
| `ENIGMA_LOG_LEVEL` | `INFO` | Logging level |
| `ENIGMA_SITUATION_TTL_MINUTES` | `30` | TTL before expired situations are removed |
| `ENIGMA_SITUATION_DORMANCY_MINUTES` | `10` | Inactivity before dormant state |
| `ENIGMA_BURST_FACTOR` | `3.0` | How much faster than average = burst |
| `ENIGMA_BURST_RECENT_COUNT` | `3` | Recent intervals to evaluate for burst |
| `ENIGMA_QUIET_WINDOW_MINUTES` | `5` | Inactivity duration that qualifies as quiet |

---

## Quick Start

```bash
pip install -e ".[dev]"
uvicorn enigma_reason.main:app --reload
```

## Run Tests

```bash
pytest -v
```
