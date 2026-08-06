"""
Module: scenarios/generator.py

Generates scenarios with injected ground truth.

A scenario is a list of signals plus a statement of what a competent reasoner
should conclude from them, including the case where the correct conclusion is
that the evidence does not support one.

Design commitments, each of which exists to avoid a specific failure the
project has already had.

Signals are produced as raw detector payloads and converted through the
existing adapters rather than constructed directly. The adapters define the
schemas and were implemented but never exercised; routing through them means a
scenario cannot drift from the format the live system accepts. The generator
inverts each adapter's normalisation to hit a target anomaly score, so the
score distribution is controlled without bypassing the adapter.

Entities are scenario scoped. Level 5 assigned entities by taking an index
modulo sixteen, which is uncorrelated with anything, so every situation was a
uniform random sample and grouping by entity grouped nothing. Here every
identifier carries its scenario tag, so grouping by entity reconstructs the
scenario. This is not label leakage: assemble_context in the reasoning graph
exposes only aggregated metrics and never sees an entity identifier. The
entity determines which signals belong together, not what they mean.

Sources vary within a scenario. A single pinned source forces three mechanisms
into a degenerate regime at once: the diversity term saturates at a third of
its weight, the sanity gate adds a low diversity boost to UNKNOWN on every
iteration, and convergence is halved whenever mean anomaly exceeds the high
anomaly threshold with diversity at or below one. A suite generated at
diversity one would measure those three penalties rather than the reasoner.

Timestamps are genuine event times with burst and quiet phases, so the Level 8
clock study can drive the same suite.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterator
from uuid import uuid4

from enigma_reason.adapters.auth import AuthAnomalyAdapter
from enigma_reason.adapters.network import NetworkAnomalyAdapter
from enigma_reason.adapters.video import VideoDetectionAdapter
from enigma_reason.domain.signal import Signal

EPOCH = datetime(2026, 3, 1, tzinfo=timezone.utc)
NEVER_SUFFICIENT = 10**9


class Regime(str, Enum):
    """The four evidence regimes a scenario can be drawn from."""

    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    SPARSE = "sparse"
    UNKNOWN_ATTACK = "unknown_attack"


class Detector(str, Enum):
    """The three detector families the adapters accept."""

    NETWORK = "network_anomaly"
    AUTH = "auth_anomaly"
    VIDEO = "video_detection"


@dataclass(frozen=True)
class Category:
    """A narrative a scenario can be about, and how to recognise it.

    Attributes:
        name: Stable identifier used in ground truth and reporting.
        detectors: Which detector families emit evidence for this narrative.
        keywords: Terms whose presence in a hypothesis marks it as this
            narrative. Matching on text is unavoidable because the reasoner
            emits free text, and its limits are stated in the scoring module.
    """

    name: str
    detectors: tuple[Detector, ...]
    keywords: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        name="data_exfiltration",
        detectors=(Detector.NETWORK,),
        keywords=("exfiltrat", "data transfer", "outbound", "upload", "leak", "volume"),
    ),
    Category(
        name="brute_force_access",
        detectors=(Detector.AUTH,),
        keywords=("brute", "credential", "password", "login", "authentication", "guess"),
    ),
    Category(
        name="reconnaissance_sweep",
        detectors=(Detector.NETWORK, Detector.VIDEO),
        keywords=("recon", "scan", "probe", "enumerat", "sweep", "discovery"),
    ),
    Category(
        name="lateral_movement",
        detectors=(Detector.NETWORK, Detector.AUTH),
        keywords=("lateral", "pivot", "spread", "propagat", "internal movement"),
    ),
    Category(
        name="physical_intrusion",
        detectors=(Detector.VIDEO,),
        keywords=("physical", "intruder", "restricted", "premises", "camera", "zone"),
    ),
    Category(
        name="service_denial",
        detectors=(Detector.NETWORK,),
        keywords=("denial", "flood", "saturat", "overload", "exhaust", "availability"),
    ),
)

CATEGORIES_BY_NAME = {category.name: category for category in CATEGORIES}

UNKNOWN_CONCLUSION = "UNKNOWN"

DETECTOR_POOL: dict[Detector, tuple[str, ...]] = {
    Detector.NETWORK: (
        "net-detector-01",
        "net-detector-02",
        "flow-monitor-a",
        "perimeter-ids",
    ),
    Detector.AUTH: (
        "auth-detector-01",
        "auth-detector-02",
        "identity-broker",
        "sso-monitor",
    ),
    Detector.VIDEO: (
        "vision-detector-01",
        "vision-detector-02",
        "cam-analytics",
        "perimeter-vision",
    ),
}


@dataclass(frozen=True)
class GroundTruth:
    """What a competent reasoner should conclude, and why.

    Attributes:
        regime: Which evidence regime the scenario was drawn from.
        expected_conclusion: The category a correct reasoner should name, or
            UNKNOWN when no conclusion is warranted.
        should_conclude: Whether concluding at all is correct.
        conclusion_keywords: Terms marking a hypothesis as the right narrative.
        competing_keywords: For ambiguous scenarios, terms marking the rival
            narrative that the evidence equally supports. Naming either one
            confidently is a false conclusion, because the evidence does not
            separate them.
        sufficient_evidence_count: The evidence count at or above which
            concluding is justified. Set beyond reach when no amount of this
            scenario's evidence justifies a confident single conclusion, which
            is what makes premature convergence measurable.
        rationale: Plain statement of why, carried into the suite so a reader
            can audit the label without rerunning the generator.
    """

    regime: Regime
    expected_conclusion: str
    should_conclude: bool
    conclusion_keywords: tuple[str, ...]
    competing_keywords: tuple[str, ...]
    sufficient_evidence_count: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        payload = asdict(self)
        payload["regime"] = self.regime.value
        payload["conclusion_keywords"] = list(self.conclusion_keywords)
        payload["competing_keywords"] = list(self.competing_keywords)
        return payload


@dataclass(frozen=True)
class ScenarioParameters:
    """The knobs that shape one scenario."""

    signal_count: int
    source_diversity: int
    entity_count: int
    anomaly_mean: float
    anomaly_spread: float
    abstained_fraction: float
    burst_phase: bool
    quiet_phase: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view."""
        return asdict(self)


@dataclass
class Scenario:
    """One generated scenario."""

    scenario_id: str
    seed: int
    ground_truth: GroundTruth
    parameters: ScenarioParameters
    entities: list[str]
    sources: list[str]
    signals: list[Signal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable view including every signal."""
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "ground_truth": self.ground_truth.to_dict(),
            "parameters": self.parameters.to_dict(),
            "entities": list(self.entities),
            "sources": list(self.sources),
            "signal_count": len(self.signals),
            "signals": [
                json.loads(signal.model_dump_json()) for signal in self.signals
            ],
        }


REGIME_PROFILE: dict[Regime, dict[str, Any]] = {
    Regime.CLEAR: {
        "signal_count": (12, 26),
        "source_diversity": (2, 4),
        "anomaly_mean": (0.72, 0.90),
        "anomaly_spread": (0.03, 0.08),
        "abstained_fraction": (0.0, 0.08),
        "sufficient_evidence_count": 6,
    },
    Regime.AMBIGUOUS: {
        "signal_count": (10, 22),
        "source_diversity": (2, 4),
        "anomaly_mean": (0.45, 0.62),
        "anomaly_spread": (0.16, 0.26),
        "abstained_fraction": (0.10, 0.28),
        "sufficient_evidence_count": NEVER_SUFFICIENT,
    },
    Regime.SPARSE: {
        "signal_count": (1, 3),
        "source_diversity": (1, 2),
        "anomaly_mean": (0.30, 0.60),
        "anomaly_spread": (0.05, 0.15),
        "abstained_fraction": (0.0, 0.35),
        "sufficient_evidence_count": NEVER_SUFFICIENT,
    },
    Regime.UNKNOWN_ATTACK: {
        "signal_count": (8, 20),
        "source_diversity": (2, 4),
        "anomaly_mean": (0.60, 0.85),
        "anomaly_spread": (0.10, 0.20),
        "abstained_fraction": (0.35, 0.70),
        "sufficient_evidence_count": NEVER_SUFFICIENT,
    },
}


class ScenarioGenerator:
    """Produces scenarios deterministically from a seed.

    Args:
        seed: Drives every random choice. Two generators built with the same
            seed emit byte identical suites.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._random = random.Random(seed)
        self._adapters = {
            Detector.NETWORK: NetworkAnomalyAdapter(),
            Detector.AUTH: AuthAnomalyAdapter(),
            Detector.VIDEO: VideoDetectionAdapter(),
        }

    def generate(self, index: int, regime: Regime) -> Scenario:
        """Build one scenario in the given regime."""
        rng = random.Random(f"{self.seed}:{index}:{regime.value}")
        profile = REGIME_PROFILE[regime]
        scenario_id = f"s{index:05d}"

        parameters = self._draw_parameters(rng, profile, regime)
        category = self._draw_category(rng, regime)
        competitor = self._draw_competitor(rng, category, regime)
        ground_truth = self._build_ground_truth(
            regime, category, competitor, profile["sufficient_evidence_count"]
        )

        detectors = self._detectors_for(category, competitor, regime, rng)
        sources_by_detector = self._draw_sources(
            rng, detectors, parameters.source_diversity
        )
        entities = self._draw_entities(rng, scenario_id, detectors, parameters.entity_count)

        signals: list[Signal] = []
        for entity in entities:
            signals.extend(
                self._build_signals(rng, entity, sources_by_detector, parameters)
            )

        all_sources = sorted(
            {source for pool in sources_by_detector.values() for source in pool}
        )

        return Scenario(
            scenario_id=scenario_id,
            seed=self.seed,
            ground_truth=ground_truth,
            parameters=parameters,
            entities=[identifier for _, identifier in entities],
            sources=all_sources,
            signals=signals,
        )

    def _draw_parameters(
        self, rng: random.Random, profile: dict[str, Any], regime: Regime
    ) -> ScenarioParameters:
        """Sample the shape knobs for one scenario."""
        low, high = profile["signal_count"]
        signal_count = rng.randint(low, high)
        diversity_low, diversity_high = profile["source_diversity"]
        return ScenarioParameters(
            signal_count=signal_count,
            source_diversity=rng.randint(diversity_low, diversity_high),
            entity_count=1 if regime is Regime.SPARSE else rng.choice([1, 1, 2, 3]),
            anomaly_mean=round(rng.uniform(*profile["anomaly_mean"]), 4),
            anomaly_spread=round(rng.uniform(*profile["anomaly_spread"]), 4),
            abstained_fraction=round(rng.uniform(*profile["abstained_fraction"]), 4),
            burst_phase=rng.random() < 0.5,
            quiet_phase=rng.random() < 0.4,
        )

    def _draw_category(self, rng: random.Random, regime: Regime) -> Category:
        """Choose the narrative the scenario is about."""
        return rng.choice(CATEGORIES)

    def _draw_competitor(
        self, rng: random.Random, category: Category, regime: Regime
    ) -> Category | None:
        """Choose the rival narrative for an ambiguous scenario."""
        if regime is not Regime.AMBIGUOUS:
            return None
        alternatives = [c for c in CATEGORIES if c.name != category.name]
        return rng.choice(alternatives)

    def _build_ground_truth(
        self,
        regime: Regime,
        category: Category,
        competitor: Category | None,
        sufficient: int,
    ) -> GroundTruth:
        """State what a correct reasoner should conclude and why."""
        if regime is Regime.CLEAR:
            return GroundTruth(
                regime=regime,
                expected_conclusion=category.name,
                should_conclude=True,
                conclusion_keywords=category.keywords,
                competing_keywords=(),
                sufficient_evidence_count=sufficient,
                rationale=(
                    "Evidence is plentiful, consistent and drawn from several "
                    f"sources, and points at {category.name} alone."
                ),
            )
        if regime is Regime.AMBIGUOUS:
            rival = competitor.name if competitor else "an unnamed alternative"
            rival_keywords = competitor.keywords if competitor else ()
            return GroundTruth(
                regime=regime,
                expected_conclusion=UNKNOWN_CONCLUSION,
                should_conclude=False,
                conclusion_keywords=category.keywords,
                competing_keywords=rival_keywords,
                sufficient_evidence_count=sufficient,
                rationale=(
                    f"Evidence fits {category.name} and {rival} equally well and "
                    "does not separate them, so a confident single conclusion is "
                    "unsupported however much of this evidence arrives."
                ),
            )
        if regime is Regime.SPARSE:
            return GroundTruth(
                regime=regime,
                expected_conclusion=UNKNOWN_CONCLUSION,
                should_conclude=False,
                conclusion_keywords=(),
                competing_keywords=(),
                sufficient_evidence_count=sufficient,
                rationale=(
                    "Too few observations from too few sources to support any "
                    "conclusion about what is happening."
                ),
            )
        return GroundTruth(
            regime=regime,
            expected_conclusion=UNKNOWN_CONCLUSION,
            should_conclude=False,
            conclusion_keywords=(),
            competing_keywords=(),
            sufficient_evidence_count=sufficient,
            rationale=(
                "Signals derive from the held out class, so no known hypothesis "
                "is the right answer and a high abstention rate is what the "
                "sensor is signalling."
            ),
        )

    def _detectors_for(
        self,
        category: Category,
        competitor: Category | None,
        regime: Regime,
        rng: random.Random,
    ) -> tuple[Detector, ...]:
        """Choose which detector families emit for this scenario."""
        detectors = set(category.detectors)
        if competitor is not None:
            detectors |= set(competitor.detectors)
        if regime is Regime.UNKNOWN_ATTACK:
            detectors |= {rng.choice(list(Detector))}
        return tuple(sorted(detectors, key=lambda d: d.value))

    def _draw_sources(
        self,
        rng: random.Random,
        detectors: tuple[Detector, ...],
        diversity: int,
    ) -> dict[Detector, list[str]]:
        """Pick the source identifiers each detector family reports from.

        Sources are drawn per family rather than from one merged pool. Drawing
        from a merged pool and then filtering to the family an entity belongs
        to left some entities with a single eligible source, which is the
        degenerate regime this parameter exists to avoid: at diversity one the
        confidence term saturates at a third of its weight, the sanity gate
        adds a low diversity boost to UNKNOWN every iteration, and convergence
        is halved once mean anomaly passes the high anomaly threshold. The
        parameter therefore means distinct sources per situation, which is the
        unit the reasoner actually sees.
        """
        return {
            detector: sorted(
                rng.sample(
                    list(DETECTOR_POOL[detector]),
                    min(diversity, len(DETECTOR_POOL[detector])),
                )
            )
            for detector in detectors
        }

    def _draw_entities(
        self,
        rng: random.Random,
        scenario_id: str,
        detectors: tuple[Detector, ...],
        entity_count: int,
    ) -> list[tuple[Detector, str]]:
        """Build scenario scoped entity identifiers.

        Every identifier embeds the scenario tag, so grouping by entity
        reconstructs the scenario rather than mixing unrelated signals as the
        Level 5 index modulo sixteen assignment did.
        """
        entities: list[tuple[Detector, str]] = []
        for position in range(entity_count):
            detector = detectors[position % len(detectors)]
            if detector is Detector.NETWORK:
                identifier = f"10.{rng.randint(8, 250)}.{position}.{rng.randint(2, 250)}-{scenario_id}"
            elif detector is Detector.AUTH:
                identifier = f"user-{scenario_id}-{position:02d}"
            else:
                identifier = f"cam-{scenario_id}-{position:02d}"
            entities.append((detector, identifier))
        return entities

    def _build_signals(
        self,
        rng: random.Random,
        entity: tuple[Detector, str],
        sources_by_detector: dict[Detector, list[str]],
        parameters: ScenarioParameters,
    ) -> list[Signal]:
        """Emit one entity's signals through the adapters."""
        detector, identifier = entity
        eligible = sources_by_detector[detector]

        signals: list[Signal] = []
        for position, offset in enumerate(
            self._arrival_offsets(rng, parameters.signal_count, parameters)
        ):
            anomaly = self._draw_anomaly(rng, parameters)
            source = eligible[position % len(eligible)]
            abstained = rng.random() < parameters.abstained_fraction
            payload = self._build_payload(
                detector, identifier, source, anomaly, EPOCH + offset, rng
            )
            signal = self._adapters[detector].adapt(payload)
            signals.append(
                signal.model_copy(
                    update={
                        "abstained": abstained,
                        "calibrated_confidence": round(
                            max(0.0, min(1.0, signal.confidence)), 4
                        ),
                    }
                )
            )
        return signals

    def _arrival_offsets(
        self,
        rng: random.Random,
        count: int,
        parameters: ScenarioParameters,
    ) -> Iterator[timedelta]:
        """Yield event time offsets, optionally with burst and quiet phases.

        Offsets are event times, not ingest times, so the Level 8 clock study
        can replay this suite in either domain and observe a difference.
        """
        elapsed = 0.0
        burst_at = rng.randint(1, max(1, count - 1)) if parameters.burst_phase else -1
        quiet_at = rng.randint(1, max(1, count - 1)) if parameters.quiet_phase else -1

        for position in range(count):
            if position == burst_at:
                gap = rng.uniform(0.2, 1.0)
            elif position == quiet_at:
                gap = rng.uniform(400.0, 900.0)
            else:
                gap = rng.uniform(8.0, 45.0)
            elapsed += gap
            yield timedelta(seconds=round(elapsed, 3))

    @staticmethod
    def _draw_anomaly(rng: random.Random, parameters: ScenarioParameters) -> float:
        """Sample one anomaly score from the scenario's distribution."""
        value = rng.gauss(parameters.anomaly_mean, parameters.anomaly_spread)
        return round(max(0.02, min(0.99, value)), 4)

    def _build_payload(
        self,
        detector: Detector,
        identifier: str,
        source: str,
        anomaly: float,
        timestamp: datetime,
        rng: random.Random,
    ) -> dict[str, Any]:
        """Construct a raw detector payload targeting a given anomaly score.

        Each branch inverts the adapter's own normalisation, so the requested
        anomaly score survives the conversion without the generator having to
        reach past the adapter and write the field directly.
        """
        if detector is Detector.NETWORK:
            return {
                "source_type": "network_anomaly",
                "src_ip": identifier,
                "dst_ip": f"203.0.113.{rng.randint(2, 250)}",
                "protocol": rng.choice(["tcp", "udp"]),
                "bytes_sent": rng.randint(600_000, 4_000_000)
                if anomaly > 0.6
                else rng.randint(1_000, 90_000),
                "bytes_received": rng.randint(200, 9_000),
                "z_score": round(anomaly * 10.0, 4),
                "detector_id": source,
                "timestamp": timestamp.isoformat(),
            }
        if detector is Detector.AUTH:
            return {
                "source_type": "auth_anomaly",
                "username": identifier,
                "failed_attempts": max(1, round(anomaly * 20.0)),
                "window_seconds": rng.choice([30, 60, 120]),
                "source_ip": f"192.168.{rng.randint(1, 250)}.{rng.randint(2, 250)}",
                "detector_id": source,
                "timestamp": timestamp.isoformat(),
            }
        return {
            "source_type": "video_detection",
            "camera_id": identifier,
            "person_detected": True,
            "object_class": rng.choice(["person", "vehicle"]),
            "confidence": anomaly,
            "zone": rng.choice(["lobby", "corridor", "car_park"]),
            "detector_id": source,
            "timestamp": timestamp.isoformat(),
        }


def generate_suite(
    total: int = 400,
    seed: int = 42,
    regimes: tuple[Regime, ...] = tuple(Regime),
) -> list[Scenario]:
    """Build a suite balanced across the regimes.

    Args:
        total: How many scenarios to produce. Rounded up so each regime
            receives the same count.
        seed: Drives every draw.
        regimes: Which regimes to include.

    Returns:
        The scenarios, ordered by regime then index.
    """
    generator = ScenarioGenerator(seed=seed)
    per_regime = -(-total // len(regimes))
    scenarios: list[Scenario] = []
    index = 0
    for regime in regimes:
        for _ in range(per_regime):
            scenarios.append(generator.generate(index, regime))
            index += 1
    return scenarios


def suite_hash(scenarios: list[Scenario]) -> str:
    """Return a content hash freezing a suite.

    Signal identifiers are excluded because they are freshly generated uuids
    and would make an otherwise identical suite hash differently. Everything
    that determines what the suite means is included.
    """
    digest = hashlib.sha256()
    for scenario in scenarios:
        digest.update(scenario.scenario_id.encode("utf-8"))
        digest.update(json.dumps(scenario.ground_truth.to_dict(), sort_keys=True).encode("utf-8"))
        digest.update(json.dumps(scenario.parameters.to_dict(), sort_keys=True).encode("utf-8"))
        digest.update(json.dumps(sorted(scenario.entities), sort_keys=True).encode("utf-8"))
        digest.update(json.dumps(sorted(scenario.sources), sort_keys=True).encode("utf-8"))
        for signal in scenario.signals:
            digest.update(
                json.dumps(
                    {
                        "timestamp": signal.timestamp.isoformat(),
                        "signal_type": signal.signal_type.value,
                        "entity": str(signal.entity) if signal.entity else "",
                        "anomaly_score": signal.anomaly_score,
                        "confidence": signal.confidence,
                        "source": signal.source,
                        "abstained": signal.abstained,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
    return digest.hexdigest()


def scenario_id_from_entity(entity: str) -> str | None:
    """Recover the scenario tag embedded in an entity identifier.

    This is what makes a situation traceable back to the ground truth that
    produced it, and is the property Level 5 lacked.
    """
    for part in str(entity).replace(":", "-").split("-"):
        if len(part) == 6 and part.startswith("s") and part[1:].isdigit():
            return part
    return None
