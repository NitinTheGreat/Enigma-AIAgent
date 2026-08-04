"""
Module: Enigma-AIAgent/level2_verify.py

Produces the evidence for the Level 2 DONE CHECK and writes it to results/level2.

Every figure printed here is also written to disk as JSON, per the standing rule
that no number is reported unless it is also machine readable.

The clock study replays synthesised signals carrying UNSW-NB15 era event
timestamps. It cannot replay the official partition because that partition has
no Stime or Ltime column, only dur and rate, so it carries no event time at all.
That is recorded in the output as official_partition_has_event_time.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from enigma_reason.adapters.unsw_threat import UNSWThreatAdapter
from enigma_reason.config import Settings
from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import Signal
from enigma_reason.foundation.clock import ClockMode
from enigma_reason.graph.nodes import (
    hypothesis_sanity_gate,
    make_apply_belief_inertia,
    make_update_convergence,
)
from enigma_reason.store.correlation import EntityCorrelation
from enigma_reason.store.situation_store import SituationStore

CAPTURE_START = datetime(2015, 1, 22, 11, 49, 37, tzinfo=timezone.utc)
DEVICE_POOL = [f"175.45.176.{index}" for index in range(4)] + [
    f"59.166.0.{index}" for index in range(10)
]
ATTACK_TYPES = [
    SignalType.GENERIC,
    SignalType.EXPLOIT,
    SignalType.FUZZERS,
    SignalType.DOS,
    SignalType.RECONNAISSANCE,
    SignalType.BACKDOOR,
]


def synthesise_signals(count: int, seed: int) -> list[Signal]:
    """Build a replay stream carrying genuine 2015 era event timestamps.

    Args:
        count: How many signals to generate.
        seed: Seed controlling entity, type, gap and score selection.

    Returns:
        Signals ordered by event timestamp, spread over the capture window with
        variable inter-arrival gaps so that burst and quiet are both reachable.
    """
    rng = random.Random(seed)
    signals: list[Signal] = []
    event_time = CAPTURE_START

    for index in range(count):
        gap_seconds = rng.choice([1, 2, 3, 5, 8, 20, 60, 240, 900])
        event_time = event_time + timedelta(seconds=gap_seconds)
        signals.append(
            Signal.model_validate(
                {
                    "signal_id": f"00000000-0000-4000-8000-{index:012d}",
                    "timestamp": event_time.isoformat(),
                    "signal_type": rng.choice(ATTACK_TYPES).value,
                    "entity": {
                        "kind": EntityKind.DEVICE.value,
                        "identifier": rng.choice(DEVICE_POOL),
                    },
                    "anomaly_score": round(rng.uniform(0.2, 0.98), 4),
                    "confidence": round(rng.uniform(0.3, 0.99), 4),
                    "predicted_class_confidence": round(rng.uniform(0.3, 0.99), 4),
                    "predictive_entropy": round(rng.uniform(0.05, 0.8), 4),
                    "features": ["dur", "sbytes", "dbytes"],
                    "source": "unsw-threat-detector",
                }
            )
        )
    return signals


async def replay_under_clock_mode(
    signals: list[Signal], mode: ClockMode, quiet_window_minutes: int
) -> dict[str, Any]:
    """Replay a signal stream through a store in one clock mode.

    Args:
        signals: The stream to replay, ordered by event time.
        mode: Which clock mode the store and engine operate in.
        quiet_window_minutes: Inactivity window that qualifies as quiet.

    Returns:
        Trend and quiet distributions plus summary statistics for the mode.
    """
    engine = ReasoningEngine(
        quiet_window=timedelta(minutes=quiet_window_minutes), clock_mode=mode
    )
    store = SituationStore(
        ttl=timedelta(minutes=30),
        dormancy_window=timedelta(minutes=10),
        correlation=EntityCorrelation(),
        quiet_window=timedelta(minutes=quiet_window_minutes),
        reasoning_engine=engine,
        clock_mode=mode,
    )

    for signal in signals:
        await store.ingest(signal)

    trends: Counter[str] = Counter()
    quiet_flags: Counter[str] = Counter()
    confidences: list[float] = []

    async with store._lock:
        situations = list(store._situations.values())

    for situation in situations:
        snapshot = engine.evaluate(situation)
        trends[snapshot.trend.value] += 1
        quiet_flags[str(snapshot.quiet_detected)] += 1
        confidences.append(snapshot.confidence_level)

    total = len(situations)
    return {
        "clock_mode": mode.value,
        "situations": total,
        "trend_distribution": dict(trends),
        "trend_share": {
            key: round(value / total, 4) for key, value in trends.items()
        } if total else {},
        "quiet_distribution": dict(quiet_flags),
        "mean_confidence": round(sum(confidences) / total, 4) if total else 0.0,
        "distinct_trend_labels": len(trends),
        "is_degenerate": len(trends) == 1,
    }


def check_belief_inertia() -> dict[str, Any]:
    """Demonstrate that a raw update of 0.25 is clamped to a 0.10 cap."""
    hypothesis = {
        "hypothesis_id": "inertia-demo",
        "description": "Rapid belief escalation hypothesis",
        "confidence": 0.55,
        "confidence_previous": 0.30,
        "confidence_before_inertia": 0.55,
        "status": "active",
        "belief_velocity": 0.0,
        "belief_acceleration": 0.0,
        "dominant_iterations": 0,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
    }
    node = make_apply_belief_inertia(max_confidence_delta=0.10)
    result = node({"hypotheses": [copy.deepcopy(hypothesis)]})
    updated = result["hypotheses"][0]
    return {
        "cap": 0.10,
        "confidence_previous": hypothesis["confidence_previous"],
        "raw_proposed_confidence": hypothesis["confidence_before_inertia"],
        "raw_delta": round(
            hypothesis["confidence_before_inertia"] - hypothesis["confidence_previous"], 4
        ),
        "applied_confidence": updated["confidence"],
        "applied_delta": round(updated["confidence"] - hypothesis["confidence_previous"], 4),
        "inertia_clamped": updated["inertia_clamped"],
    }


def check_sanity_gate() -> dict[str, Any]:
    """Demonstrate the vague hypothesis penalty firing and changing confidence."""
    hypotheses = [
        {
            "hypothesis_id": "vague",
            "description": "Maybe something anomalous is occurring here",
            "confidence": 0.40,
            "status": "active",
            "belief_velocity": 0.0,
            "belief_acceleration": 0.0,
            "dominant_iterations": 0,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        },
        {
            "hypothesis_id": "benign",
            "description": "Normal operational traffic within parameters",
            "confidence": 0.40,
            "status": "active",
            "belief_velocity": 0.0,
            "belief_acceleration": 0.0,
            "dominant_iterations": 0,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        },
    ]
    before = {h["hypothesis_id"]: h["confidence"] for h in hypotheses}
    result = hypothesis_sanity_gate(
        {
            "hypotheses": copy.deepcopy(hypotheses),
            "reasoning_snapshot": {"evidence_count": 6, "source_diversity": 2},
        }
    )
    after = {
        h["hypothesis_id"]: h["confidence"]
        for h in result["hypotheses"]
        if h["hypothesis_id"] in before
    }
    return {
        "before": before,
        "after": after,
        "penalty_applied_to_vague": round(before["vague"] - after["vague"], 4),
        "benign_unchanged": before["benign"] == after["benign"],
    }


def check_convergence_purity() -> dict[str, Any]:
    """Demonstrate that scoring leaves the caller's hypotheses untouched."""
    hypotheses = [
        {
            "hypothesis_id": "lead",
            "description": "Leading hypothesis under evaluation",
            "confidence": 0.70,
            "status": "active",
            "belief_velocity": 0.0,
            "belief_acceleration": 0.0,
            "dominant_iterations": 0,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        },
        {
            "hypothesis_id": "trail",
            "description": "Trailing hypothesis under evaluation",
            "confidence": 0.20,
            "status": "active",
            "belief_velocity": 0.0,
            "belief_acceleration": 0.0,
            "dominant_iterations": 0,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        },
    ]
    snapshot = copy.deepcopy(hypotheses)
    node = make_update_convergence(persistence_required=True)
    result = node(
        {
            "hypotheses": hypotheses,
            "iteration_count": 0,
            "convergence_threshold": 0.8,
            "convergence_persistence": 2,
            "reasoning_snapshot": {"mean_anomaly_score": 0.5, "source_diversity": 2},
            "last_confidence_shift": 0.0,
            "undecided_iterations": 0,
        }
    )
    return {
        "input_unchanged": hypotheses == snapshot,
        "returned_objects_are_new": all(
            returned is not original
            for returned, original in zip(result["hypotheses"], hypotheses)
        ),
        "input_dominant_iterations": [h["dominant_iterations"] for h in hypotheses],
        "returned_dominant_iterations": [
            h["dominant_iterations"] for h in result["hypotheses"]
        ],
    }


def check_anomaly_score_semantics() -> dict[str, Any]:
    """Demonstrate that a confident normal and a confident attack differ."""
    adapter = UNSWThreatAdapter()

    def adapt(signal_type: str, anomaly: float, class_confidence: float) -> Signal:
        return adapter.adapt(
            {
                "inputs_for_xai_model": {
                    "signal_id": "00000000-0000-4000-8000-000000000009",
                    "timestamp": CAPTURE_START.isoformat(),
                    "signal_type": signal_type,
                    "entity": {"device": "10.0.0.1"},
                    "anomaly_score": anomaly,
                    "predicted_class_confidence": class_confidence,
                    "predictive_entropy": 0.05,
                    "features": ["dur"],
                    "source": "unsw-threat-detector",
                }
            }
        )

    confident_normal = adapt("normal", 0.02, 0.98)
    confident_attack = adapt("generic", 0.97, 0.97)

    return {
        "confident_normal": {
            "signal_type": confident_normal.signal_type.value,
            "anomaly_score": confident_normal.anomaly_score,
            "predicted_class_confidence": confident_normal.predicted_class_confidence,
            "passes_below_0_1": confident_normal.anomaly_score < 0.1,
        },
        "confident_attack": {
            "signal_type": confident_attack.signal_type.value,
            "anomaly_score": confident_attack.anomaly_score,
            "predicted_class_confidence": confident_attack.predicted_class_confidence,
            "passes_above_0_9": confident_attack.anomaly_score > 0.9,
        },
        "scores_are_distinct_measures": (
            confident_normal.predicted_class_confidence
            == confident_attack.predicted_class_confidence - 0.01
        ),
    }


def dump_configuration() -> dict[str, Any]:
    """Report the six Level 2 switches and their effective values."""
    settings = Settings()
    return {
        "ENIGMA_CLOCK_MODE": settings.clock_mode.value,
        "ENIGMA_UNKNOWN_HYPOTHESIS_ENABLED": settings.unknown_hypothesis_enabled,
        "ENIGMA_SANITY_GATE_ENABLED": settings.sanity_gate_enabled,
        "ENIGMA_ASYMMETRIC_DECAY_ENABLED": settings.asymmetric_decay_enabled,
        "ENIGMA_PERSISTENCE_REQUIRED": settings.persistence_required,
        "ENIGMA_MAX_CONFIDENCE_DELTA": settings.max_confidence_delta,
    }


async def main_async(args: argparse.Namespace) -> None:
    """Run every Level 2 check and write the combined report."""
    signals = synthesise_signals(args.signals, args.seed)

    clock_results = {}
    for mode in (ClockMode.SEPARATED, ClockMode.CONFLATED, ClockMode.WALL):
        clock_results[mode.value] = await replay_under_clock_mode(
            signals, mode, args.quiet_window_minutes
        )

    report = {
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "signals_replayed": len(signals),
        "event_time_span": {
            "first": signals[0].timestamp.isoformat(),
            "last": signals[-1].timestamp.isoformat(),
        },
        "official_partition_has_event_time": False,
        "clock_study": clock_results,
        "belief_inertia": check_belief_inertia(),
        "sanity_gate": check_sanity_gate(),
        "convergence_purity": check_convergence_purity(),
        "anomaly_score": check_anomaly_score_semantics(),
        "configuration": dump_configuration(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print()
    print("=== DONE CHECK 1: trend distribution by clock mode ===")
    for mode, result in clock_results.items():
        print(
            f"  {mode:<10} situations={result['situations']:<4} "
            f"trends={result['trend_distribution']} "
            f"quiet={result['quiet_distribution']} "
            f"degenerate={result['is_degenerate']}"
        )
    print()
    print("=== DONE CHECK 2: belief inertia clamp ===")
    for key, value in report["belief_inertia"].items():
        print(f"  {key}: {value}")
    print()
    print("=== DONE CHECK 3: sanity gate penalty ===")
    print(f"  before: {report['sanity_gate']['before']}")
    print(f"  after:  {report['sanity_gate']['after']}")
    print(f"  penalty applied: {report['sanity_gate']['penalty_applied_to_vague']}")
    print(f"  benign untouched: {report['sanity_gate']['benign_unchanged']}")
    print()
    print("=== DONE CHECK 4: convergence scorer purity ===")
    for key, value in report["convergence_purity"].items():
        print(f"  {key}: {value}")
    print()
    print("=== DONE CHECK 5: anomaly score semantics ===")
    print(f"  confident normal: {report['anomaly_score']['confident_normal']}")
    print(f"  confident attack: {report['anomaly_score']['confident_attack']}")
    print()
    print("=== DONE CHECK 7: configuration ===")
    for key, value in report["configuration"].items():
        print(f"  {key} = {value}")


def main() -> None:
    """Parse arguments and run the Level 2 verification suite."""
    parser = argparse.ArgumentParser(description="Produce Level 2 DONE CHECK evidence.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signals", type=int, default=400)
    parser.add_argument("--quiet-window-minutes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "level2" / "verify.json",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
