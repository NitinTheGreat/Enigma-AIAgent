"""
Module: Enigma-AIAgent/level5_reasoning_baselines.py

Stage two of the reasoning-layer baselines. Reads the signal stream exported by
the sensor and compares three ways of consuming it.

    flat alerting     every forwarded detection becomes its own alert, with no
                      grouping, no temporal analysis and no reasoning. This is
                      the baseline the architecture argues against.
    rule only         situation grouping plus the deterministic temporal and
                      confidence layer, with the LangGraph and LLM loop removed
                      entirely. Output is the reasoning snapshot alone.
    full system       adds LLM hypothesis generation. Not run here, see L5.9.

Ground truth caveat, stated once and carried into every number below. A
situation has no true label. The reference used here is the majority
ground-truth attack_cat of the signals the correlator grouped together. That is
an approximation and it is circular in one direction: if grouping is wrong, the
reference it produces is also wrong. Level 7's scenario generator supplies real
situation-level ground truth. Nothing here should be reported as situation
accuracy without that caveat attached.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from enigma_reason.adapters.unsw_threat import _SIGNAL_TYPE_MAP
from enigma_reason.core.reasoning_engine import ReasoningEngine
from enigma_reason.domain.enums import EntityKind, SignalType
from enigma_reason.domain.signal import Signal
from enigma_reason.foundation.clock import ClockMode
from enigma_reason.store.correlation import EntityCorrelation
from enigma_reason.store.situation_store import SituationStore


def normalise_label(raw: str) -> str:
    """Map a label to the canonical signal type value.

    Ground truth carries dataset spellings such as Exploits and DoS while the
    sensor emits enum values such as exploit and dos. Both sides go through the
    adapter's own mapping so the comparison cannot be thrown by casing or
    pluralisation.

    Args:
        raw: A label from either side.

    Returns:
        The canonical signal type value, or unknown.
    """
    key = str(raw).strip().lower()
    mapped = _SIGNAL_TYPE_MAP.get(key)
    if mapped is not None:
        return mapped.value
    return SignalType.UNKNOWN.value


def load_signals(path: Path) -> list[dict]:
    """Read the exported signal stream."""
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_signal(entry: dict) -> Signal:
    """Turn one exported record into a validated Signal."""
    return Signal.model_validate(
        {
            "signal_id": entry["signal_id"],
            "timestamp": entry["timestamp"],
            "signal_type": normalise_label(entry["emitted_signal_type"])
            if entry["emitted_signal_type"] != "unknown"
            else SignalType.UNKNOWN.value,
            "entity": {
                "kind": EntityKind.DEVICE.value,
                "identifier": entry["entity_device"],
            },
            "anomaly_score": entry["anomaly_score"],
            "confidence": entry["predicted_class_confidence"],
            "abstained": entry["abstained"],
            "calibrated_confidence": entry["calibrated_confidence"],
            "features": ["dur", "sbytes", "dbytes"],
            "source": entry["source"],
        }
    )


def measure_flat_alerting(entries: list[dict], interval_seconds: float) -> dict:
    """Count what a system with no grouping would emit.

    Args:
        entries: The exported signal stream.
        interval_seconds: Replay spacing, used to derive a rate per minute.

    Returns:
        Alert volume, rate and false alert rate.
    """
    forwarded = [entry for entry in entries if entry["routed_downstream"]]
    false_alerts = [entry for entry in forwarded if entry["ground_truth_is_normal"]]
    span_seconds = max(len(entries) * interval_seconds, 1.0)

    return {
        "records_replayed": len(entries),
        "alerts": len(forwarded),
        "alerts_per_minute": round(len(forwarded) / (span_seconds / 60.0), 4),
        "replay_span_seconds": round(span_seconds, 2),
        "false_alerts": len(false_alerts),
        "false_alert_rate": round(len(false_alerts) / len(forwarded), 6) if forwarded else 0.0,
        "abstained_alerts": sum(1 for entry in forwarded if entry["abstained"]),
    }


async def measure_situation_grouping(
    entries: list[dict], quiet_window_minutes: int, ttl_minutes: int
) -> dict:
    """Group the same stream into situations and score the deterministic layer.

    Args:
        entries: The exported signal stream.
        quiet_window_minutes: Inactivity window for the temporal layer.
        ttl_minutes: Situation time to live.

    Returns:
        Grouping counts, compression ratio and rule-only correctness.
    """
    engine = ReasoningEngine(
        quiet_window=timedelta(minutes=quiet_window_minutes),
        clock_mode=ClockMode.SEPARATED,
    )
    store = SituationStore(
        ttl=timedelta(minutes=ttl_minutes),
        dormancy_window=timedelta(minutes=ttl_minutes // 3),
        correlation=EntityCorrelation(),
        quiet_window=timedelta(minutes=quiet_window_minutes),
        reasoning_engine=engine,
        clock_mode=ClockMode.SEPARATED,
    )

    forwarded = [entry for entry in entries if entry["routed_downstream"]]
    truth_by_situation: dict[str, list[str]] = {}
    emitted_by_situation: dict[str, list[str]] = {}

    for entry in forwarded:
        signal = build_signal(entry)
        situation = await store.ingest(signal)
        key = str(situation.situation_id)
        truth_by_situation.setdefault(key, []).append(
            normalise_label(entry["ground_truth_attack_cat"])
        )
        emitted_by_situation.setdefault(key, []).append(
            normalise_label(entry["emitted_signal_type"])
        )

    async with store._lock:
        situations = list(store._situations.values())

    correct = 0
    scored = 0
    per_situation = []
    trend_counts: Counter[str] = Counter()

    for situation in situations:
        key = str(situation.situation_id)
        truths = truth_by_situation.get(key, [])
        emitted = emitted_by_situation.get(key, [])
        if not truths or not emitted:
            continue

        snapshot = engine.evaluate(situation)
        trend_counts[snapshot.trend.value] += 1

        majority_truth = Counter(truths).most_common(1)[0][0]
        dominant_emitted = Counter(emitted).most_common(1)[0][0]
        is_correct = majority_truth == dominant_emitted
        correct += int(is_correct)
        scored += 1

        per_situation.append(
            {
                "situation_id": key,
                "evidence_count": situation.evidence_count,
                "majority_ground_truth": majority_truth,
                "dominant_emitted_type": dominant_emitted,
                "correct": is_correct,
                "confidence_level": snapshot.confidence_level,
                "trend": snapshot.trend.value,
                "abstained_fraction": snapshot.abstained_fraction,
                "ground_truth_purity": round(
                    Counter(truths).most_common(1)[0][1] / len(truths), 4
                ),
            }
        )

    signal_level_correct = sum(
        1
        for entry in forwarded
        if normalise_label(entry["ground_truth_attack_cat"])
        == normalise_label(entry["emitted_signal_type"])
    )

    return {
        "alerts_without_grouping": len(forwarded),
        "situations_created": len(situations),
        "situations_scored": scored,
        "compression_ratio": (
            round(len(forwarded) / len(situations), 4) if situations else None
        ),
        "rule_only_dominant_type_accuracy": (
            round(correct / scored, 6) if scored else None
        ),
        "rule_only_correct": correct,
        "signal_level_accuracy": (
            round(signal_level_correct / len(forwarded), 6) if forwarded else None
        ),
        "mean_ground_truth_purity": (
            round(
                sum(item["ground_truth_purity"] for item in per_situation) / len(per_situation),
                4,
            )
            if per_situation
            else None
        ),
        "trend_distribution": dict(trend_counts),
        "per_situation": per_situation,
    }


async def main_async(args: argparse.Namespace) -> None:
    """Run both reasoning baselines and write the reports."""
    entries = load_signals(args.signals)
    manifest_path = args.signals.with_name("replay_manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    interval = manifest.get("replay_interval_seconds", 0.5)

    flat = measure_flat_alerting(entries, interval)
    grouped = await measure_situation_grouping(
        entries, args.quiet_window_minutes, args.ttl_minutes
    )

    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ground_truth_caveat = (
        "A situation has no true label. The reference is the majority "
        "ground-truth attack_cat of the signals the correlator grouped together, "
        "which is an approximation and is circular if grouping is wrong. Level 7 "
        "supplies real situation-level ground truth."
    )

    alert_report = {
        "seed": args.seed,
        "generated_at_utc": generated,
        "source_manifest": manifest,
        "flat_alerting": flat,
        "situation_grouping": {
            key: value for key, value in grouped.items() if key != "per_situation"
        },
        "interpretation": (
            "Compression ratio is alerts divided by situations. It measures how "
            "much triage volume the grouping removes, not whether the grouping is "
            "correct."
        ),
    }
    args.alert_output.parent.mkdir(parents=True, exist_ok=True)
    args.alert_output.write_text(json.dumps(alert_report, indent=2), encoding="utf-8")

    rule_report = {
        "seed": args.seed,
        "generated_at_utc": generated,
        "variant": "rule only, LangGraph and LLM removed",
        "ground_truth_caveat": ground_truth_caveat,
        "situations_scored": grouped["situations_scored"],
        "rule_only_dominant_type_accuracy": grouped["rule_only_dominant_type_accuracy"],
        "signal_level_accuracy": grouped["signal_level_accuracy"],
        "mean_ground_truth_purity": grouped["mean_ground_truth_purity"],
        "trend_distribution": grouped["trend_distribution"],
        "per_situation": grouped["per_situation"],
    }
    args.rule_output.write_text(json.dumps(rule_report, indent=2), encoding="utf-8")

    print(f"wrote {args.alert_output}")
    print(f"wrote {args.rule_output}")
    print()
    print("=== flat alerting, no grouping ===")
    for key, value in flat.items():
        print(f"  {key}: {value}")
    print()
    print("=== situation grouping ===")
    for key, value in grouped.items():
        if key == "per_situation":
            continue
        print(f"  {key}: {value}")
    print()
    print("=== rule only variant ===")
    print(f"  situations scored                 : {grouped['situations_scored']}")
    print(f"  dominant type accuracy            : {grouped['rule_only_dominant_type_accuracy']}")
    print(f"  signal level accuracy for contrast: {grouped['signal_level_accuracy']}")
    print(f"  mean ground truth purity          : {grouped['mean_ground_truth_purity']}")
    print(f"  caveat: {ground_truth_caveat}")


def main() -> None:
    """Parse arguments and run the reasoning baselines."""
    parser = argparse.ArgumentParser(description="Level 5 reasoning-layer baselines.")
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("/mnt/f/XAI Project/results/baselines/replay_signals.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet-window-minutes", type=int, default=5)
    parser.add_argument("--ttl-minutes", type=int, default=30)
    parser.add_argument(
        "--alert-output",
        type=Path,
        default=Path("/mnt/f/XAI Project/results/baselines/alert_volume_comparison.json"),
    )
    parser.add_argument(
        "--rule-output",
        type=Path,
        default=Path("/mnt/f/XAI Project/results/baselines/rule_only_comparison.json"),
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
