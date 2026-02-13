# enigma-reason

**Phase 1 — Situation Memory & Signal Grounding**

Receives structured anomaly signals over WebSocket, validates them, and organises
them into long-lived Situations (evidence containers with lifecycle-aware expiry).

---

## Architecture

```
ML Service ──ws──▶ /ws/signal ──▶ SituationStore ──▶ Situation(evidence[])
```

### Signal Flow

1. ML service connects to `/ws/signal` via WebSocket
2. Each JSON payload is validated into an immutable `Signal` model
3. The `SituationStore` groups signals using a pluggable `CorrelationStrategy` (default: `signal_type + entity`)
4. Evidence is attached to the matched (or newly created) `Situation`
5. Ack is returned: `{ status, situation_id, evidence_count }`

### Situation Lifecycle

```
active ──(no evidence for dormancy window)──▶ dormant ──(no evidence for TTL)──▶ expired (removed)
   ▲                                              │
   └──────── (new evidence arrives) ◀─────────────┘
```

- **Active**: receiving evidence recently
- **Dormant**: no evidence within `ENIGMA_SITUATION_DORMANCY_MINUTES` (default 10), still retained
- **Expired**: no evidence within `ENIGMA_SITUATION_TTL_MINUTES` (default 30), eligible for removal

Dormant situations reactivate when new evidence arrives.

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

**Validation errors** return:

```json
{ "status": "error", "detail": [...] }
```

### `GET /health`

Returns system status:

```json
{ "status": "ok", "phase": 1, "active_situations": 5, "dormant_situations": 2 }
```

---

## Project Structure

```
enigma_reason/
├── main.py                          # FastAPI entry point
├── config.py                        # Env-based settings (ENIGMA_ prefix)
├── api/                             # WebSocket transport endpoints
│   └── ws_signal.py
├── domain/                          # Canonical models: Signal, Situation, enums
│   ├── enums.py
│   ├── signal.py
│   └── situation.py
├── store/                           # In-memory situation state management
│   ├── correlation.py               # Pluggable correlation strategy
│   └── situation_store.py
├── foundation/                      # Utilities: clock, ID generation
│   ├── clock.py
│   └── identifiers.py
├── adapters/                        # Signal normalization (reserved, not wired)
│   └── base.py
tests/
├── test_signal.py
├── test_situation.py
└── test_store.py
```

| Folder | Purpose |
|---|---|
| `api/` | WebSocket transport — validates at boundary, delegates to store |
| `domain/` | Canonical models: Signal (immutable contract), Situation (evidence container) |
| `store/` | In-memory state with async locking, TTL expiry, pluggable correlation |
| `foundation/` | Clock and ID generation — single patchable source of truth |
| `adapters/` | Reserved for signal normalization adapters (not integrated yet) |

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ENIGMA_APP_NAME` | `enigma-reason` | Application name |
| `ENIGMA_DEBUG` | `false` | Debug mode |
| `ENIGMA_LOG_LEVEL` | `INFO` | Logging level |
| `ENIGMA_SITUATION_TTL_MINUTES` | `30` | Time before expired situations are removed |
| `ENIGMA_SITUATION_DORMANCY_MINUTES` | `10` | Time before inactive situations go dormant |

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
