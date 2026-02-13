# enigma-reason

**Situation Memory · Temporal Awareness · Signal Adapters**

Receives structured or raw anomaly signals over WebSocket, validates/normalises
them via pluggable adapters, organises them into long-lived Situations, and
exposes temporal observations about evidence arrival patterns.

---

## Architecture

```
ML Sources ──ws──▶ /ws/raw-signal ──▶ AdapterRegistry ──▶ Signal ──▶ SituationStore
                   /ws/signal ─────────────────────────────────────▶ SituationStore
```

### Signal Flow (raw adapter path)

1. ML source connects to `/ws/raw-signal` via WebSocket
2. Each JSON payload is routed through the `AdapterRegistry`
3. The first adapter whose `can_handle()` returns True translates the payload
4. A canonical, validated `Signal` is produced
5. The Signal is ingested into the `SituationStore` (same as `/ws/signal` path)
6. Ack is returned with adapter name, situation_id, evidence_count

### Registered Adapters

| Adapter | `source_type` | Maps to Signal |
|---|---|---|
| `NetworkAnomalyAdapter` | `network_anomaly` | src_ip → device entity, z_score → anomaly_score |
| `AuthAnomalyAdapter` | `auth_anomaly` | username → user entity, failed_attempts → anomaly_score |
| `VideoDetectionAdapter` | `video_detection` | camera_id → device entity, confidence + zone → anomaly_score |

### Situation Lifecycle

```
active ──(dormancy window)──▶ dormant ──(TTL)──▶ expired (removed)
   ▲                              │
   └──── new evidence ◀───────────┘
```

---

## Endpoints

### `WebSocket /ws/signal` (canonical)

Ingests pre-validated signals. No adapter layer.

### `WebSocket /ws/raw-signal` (adapter-driven)

Ingests raw ML payloads. Routes through adapter registry.

**Request** (example network anomaly):

```json
{
  "source_type": "network_anomaly",
  "src_ip": "10.0.0.42",
  "dst_ip": "203.0.113.5",
  "protocol": "tcp",
  "bytes_sent": 1048576,
  "bytes_received": 256,
  "z_score": 4.2,
  "detector_id": "net-detector-01",
  "timestamp": "2026-02-13T14:00:00Z"
}
```

**Success response**:

```json
{ "status": "accepted", "adapter": "net-detector-01", "situation_id": "uuid", "evidence_count": 3 }
```

**No adapter match**:

```json
{ "status": "error", "reason": "no_adapter", "detail": "..." }
```

**Adaptation failure**:

```json
{ "status": "error", "reason": "adaptation_failed", "adapter": "network_anomaly", "detail": "..." }
```

### `GET /health`

```json
{
  "status": "ok",
  "phase": 3,
  "active_situations": 5,
  "dormant_situations": 2,
  "bursting_situations": 1,
  "quiet_situations": 3,
  "max_event_rate": 12.5,
  "adapters": [
    { "adapter_name": "network_anomaly", "accepted_count": 10, "rejected_count": 1 },
    { "adapter_name": "auth_anomaly", "accepted_count": 5, "rejected_count": 0 },
    { "adapter_name": "video_detection", "accepted_count": 3, "rejected_count": 0 }
  ],
  "total_adapted": 18,
  "total_rejected": 1
}
```

---

## Project Structure

```
enigma_reason/
├── main.py
├── config.py
├── api/
│   ├── ws_signal.py                 # Canonical signal endpoint
│   └── ws_raw_signal.py             # Adapter-driven raw signal endpoint
├── domain/
│   ├── enums.py
│   ├── signal.py
│   ├── situation.py
│   └── temporal.py
├── store/
│   ├── correlation.py
│   └── situation_store.py
├── adapters/
│   ├── base.py                      # SignalAdapter ABC
│   ├── registry.py                  # AdapterRegistry + stats + errors
│   ├── network.py                   # NetworkAnomalyAdapter
│   ├── auth.py                      # AuthAnomalyAdapter
│   └── video.py                     # VideoDetectionAdapter
├── foundation/
│   ├── clock.py
│   └── identifiers.py
tests/
├── test_signal.py
├── test_situation.py
├── test_store.py
├── test_temporal.py
└── test_adapters.py
```

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ENIGMA_APP_NAME` | `enigma-reason` | Application name |
| `ENIGMA_DEBUG` | `false` | Debug mode |
| `ENIGMA_LOG_LEVEL` | `INFO` | Logging level |
| `ENIGMA_SITUATION_TTL_MINUTES` | `30` | TTL before expired situations removed |
| `ENIGMA_SITUATION_DORMANCY_MINUTES` | `10` | Inactivity before dormant state |
| `ENIGMA_BURST_FACTOR` | `3.0` | Burst detection threshold multiplier |
| `ENIGMA_BURST_RECENT_COUNT` | `3` | Recent intervals for burst check |
| `ENIGMA_QUIET_WINDOW_MINUTES` | `5` | Inactivity for quiet detection |

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
