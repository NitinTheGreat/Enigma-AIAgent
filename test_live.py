"""Live test: connect to /ws/dashboard, send signals, see analysis pushed in real time."""

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

            # Explanation sections
            print("\n  --- EXPLANATION ---")
            expl = data.get("explanation", {})
            print(f"  Undecided: {expl.get('undecided')}")
            for sec in expl.get("sections", []):
                print(f"\n  [{sec['type']}] {sec['title']}")
                for b in sec.get("bullets", []):
                    print(f"    * {b}")
                if sec.get("contribution_score") is not None:
                    print(f"    Score: {sec['contribution_score']} ({sec['contribution_direction']})")
                if sec.get("counterfactuals"):
                    for cf in sec["counterfactuals"]:
                        print(f"    > IF: {cf['missing_condition']}")
                        print(f"      THEN: {cf['expected_effect']} ({cf['confidence_delta']:+.2f})")

            te = expl.get("temporal_evolution")
            if te:
                print(f"\n  [TEMPORAL] trend={te['confidence_trend']}, velocity={te['velocity']}, stability={te['stability']}")

            # Human-readable
            print("\n  --- HUMAN READABLE ---")
            print(data.get("human_readable", "N/A"))
            print("=" * 70)
            print()


async def send_signals():
    """Send 3 test signals to /ws/signal."""
    async with websockets.connect(SIGNAL_URI) as ws:
        for sig_type, score, source, features in [
            ("intrusion", 0.85, "network-ids", ["port_scan", "unusual_traffic"]),
            ("privilege_escalation", 0.92, "auth-monitor", ["sudo_abuse", "new_admin"]),
            ("data_exfiltration", 0.78, "dlp-sensor", ["large_outbound", "unusual_dest"]),
        ]:
            signal = {
                "signal_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_type": sig_type,
                "entity": {"kind": "device", "identifier": "server-prod-01"},
                "anomaly_score": score,
                "confidence": 0.75,
                "features": features,
                "source": source,
            }
            await ws.send(json.dumps(signal))
            resp = json.loads(await ws.recv())
            print(f"[SIGNAL] Sent {sig_type} -> situation={resp['situation_id']}, evidence={resp['evidence_count']}")
            await asyncio.sleep(0.5)  # Small delay between signals


async def main():
    print("Connecting to dashboard WebSocket...")
    ready = asyncio.Event()

    # Start dashboard listener in background
    listener_task = asyncio.create_task(dashboard_listener(ready))

    # Wait for dashboard connection
    await ready.wait()

    # Send signals
    print("\nSending test signals...\n")
    await send_signals()

    # Wait for analyses to arrive (Gemini calls take a few seconds)
    print("\nWaiting for analysis results (LangGraph + Gemini may take 10-30s)...\n")
    await asyncio.sleep(60)

    listener_task.cancel()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
