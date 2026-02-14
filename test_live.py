"""Live test: connect to /ws/dashboard, send signals in ML format, see analysis."""

import asyncio
import json
import uuid
from datetime import datetime, timezone

import websockets


EC2 = "13.233.93.2"
SIGNAL_URI = f"ws://{EC2}:8000/ws/signal"
DASHBOARD_URI = f"ws://{EC2}:8000/ws/dashboard"


async def dashboard_listener(ready_event: asyncio.Event):
    """Connect to /ws/dashboard and print whatever the AI layer pushes."""
    async with websockets.connect(DASHBOARD_URI) as ws:
        print("[DASHBOARD] Connected — waiting for analysis pushes...\n")
        ready_event.set()

        while True:
            raw = await ws.recv()
            data = json.loads(raw)

            print("=" * 70)
            print("[DASHBOARD] ANALYSIS RECEIVED")
            print("=" * 70)

            # Situation
            sit = data.get("situation", {})
            print(f"\n  Situation: {sit.get('situation_id')}")
            print(f"  Evidence:  {sit.get('evidence_count')} signals")
            print(f"  Types:     {sit.get('signal_types')}")
            print(f"  Entities:  {sit.get('entities')}")

            # LangGraph
            lg = data.get("langgraph", {})
            print(f"\n  LangGraph: {lg.get('iterations')} iterations, convergence={lg.get('convergence_score')}")
            for h in lg.get("hypotheses", []):
                print(f"    [{h.get('status')}] {h.get('description')} (conf={h.get('confidence', 0):.2f})")

            # Human-readable
            print("\n  --- HUMAN READABLE ---")
            print(data.get("human_readable", "N/A"))
            print("=" * 70)
            print()


async def send_signals():
    """Send test signals in the exact ML output format."""
    async with websockets.connect(SIGNAL_URI) as ws:
        # These mimic the actual ML layer output
        ml_signals = [
            {
                "input_that_i_gave_to_the_model": {
                    "dur": 9e-06, "sbytes": 200.0, "dbytes": 0.0,
                    "sloss": 0.0, "Sload": 540740.75, "Dload": 0.0,
                },
                "raw_output_from_model": [0.34, 0.42, 0.0002, 0.12, 0.02, 0.04, 0.03, 0.0002, 0.03, 0.002, 0.0002],
                "output_from_model": "⚠️ THREAT DETECTED: backdoor (Confidence: 0.42)",
                "inputs_for_xai_model": {
                    "signal_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal_type": "backdoor",
                    "entity": {"device": "server-prod-01", "user": "network_admin", "location": "server_rack_1"},
                    "anomaly_score": 0.42,
                    "confidence": 0.42,
                    "features": ["dur", "sbytes", "dbytes", "sloss", "Sload", "Dload"],
                    "source": "unsw-threat-detector",
                },
            },
            {
                "input_that_i_gave_to_the_model": {
                    "dur": 0.003, "sbytes": 50000.0, "dbytes": 100.0,
                },
                "raw_output_from_model": [0.05, 0.08, 0.02, 0.01, 0.02, 0.01, 0.75, 0.01, 0.02, 0.01, 0.02],
                "output_from_model": "⚠️ THREAT DETECTED: shellcode (Confidence: 0.75)",
                "inputs_for_xai_model": {
                    "signal_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal_type": "shellcode",
                    "entity": {"device": "server-prod-01", "user": "root", "location": "server_rack_1"},
                    "anomaly_score": 0.75,
                    "confidence": 0.75,
                    "features": ["dur", "sbytes", "dbytes", "smeansz", "dmeansz"],
                    "source": "unsw-threat-detector",
                },
            },
            {
                "input_that_i_gave_to_the_model": {
                    "dur": 0.0001, "sbytes": 800000.0, "dbytes": 50.0,
                },
                "raw_output_from_model": [0.02, 0.03, 0.85, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01],
                "output_from_model": "⚠️ THREAT DETECTED: exploit (Confidence: 0.85)",
                "inputs_for_xai_model": {
                    "signal_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal_type": "exploit",
                    "entity": {"device": "server-prod-01", "user": "www-data", "location": "dmz"},
                    "anomaly_score": 0.85,
                    "confidence": 0.85,
                    "features": ["dur", "sbytes", "dbytes", "Sload", "ct_srv_src"],
                    "source": "unsw-threat-detector",
                },
            },
        ]

        for ml_output in ml_signals:
            xai = ml_output["inputs_for_xai_model"]
            await ws.send(json.dumps(ml_output))
            resp = json.loads(await ws.recv())
            print(f"[SIGNAL] Sent {xai['signal_type']} -> {resp}")
            await asyncio.sleep(0.5)


async def main():
    print("Connecting to dashboard WebSocket...")
    ready = asyncio.Event()

    listener_task = asyncio.create_task(dashboard_listener(ready))
    await ready.wait()

    print("\nSending test signals (ML format)...\n")
    await send_signals()

    print("\nWaiting for analysis results (Gemini takes 5-15s per signal)...\n")
    await asyncio.sleep(60)

    listener_task.cancel()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
