"""
Module: scenarios/semantic.py

Matches a free text hypothesis to a narrative category by sentence embedding.

Why this exists. The reasoner emits free text and the ground truth names a
narrative, so scoring has to decide whether one describes the other. The
original scorer did this by substring search over a keyword list, which
matches only when the model happens to reach for the generator's vocabulary.
It is also capable of the opposite error: the phrase "internal data transfer"
contains the data exfiltration keyword "data transfer" while describing
something that is not exfiltration at all.

The mechanism. Each category carries a prose narrative describing its
defining action. Both the narrative and the hypothesis are embedded with a
sentence transformer that runs offline after its first download, and the
similarity between them is a cosine. A hypothesis is assigned to the category
whose narrative it is closest to, provided that closeness clears a threshold
fitted against hand labels; otherwise it is assigned to no category.

The narratives were written from the Category definitions in generator.py,
their names, their keyword lists and the detector families that feed them.
They were not written by reading what any model produced, because a narrative
tuned to observed output would flatter the scorer that uses it.

What this does not do. It does not judge whether a hypothesis is a good
explanation of the evidence, only which narrative it describes. A hypothesis
can be assigned to the right category and still be wrong about the situation.
Judging explanation quality is the Level 10 judge panel's task, and this
module deliberately does not attempt it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.27

CATEGORY_NARRATIVES: dict[str, str] = {
    "data_exfiltration": (
        "Sensitive data is being transferred out of the network to an external "
        "destination, appearing as unusually large or sustained outbound "
        "volume, uploads, or a leak of information leaving the estate."
    ),
    "brute_force_access": (
        "Repeated authentication attempts are being made against accounts, "
        "guessing passwords or credentials in sequence until a login succeeds."
    ),
    "reconnaissance_sweep": (
        "An actor is scanning and probing the estate to discover hosts, ports "
        "and services, enumerating and sweeping what is present in order to "
        "map it before attacking."
    ),
    "lateral_movement": (
        "An actor already inside the network is moving from one internal "
        "system to another, pivoting between hosts and propagating its access "
        "further into the estate."
    ),
    "physical_intrusion": (
        "An unauthorised person is physically present where they should not "
        "be, detected on camera entering restricted premises or a controlled "
        "zone."
    ),
    "service_denial": (
        "A service is being flooded with traffic or requests in order to "
        "saturate and exhaust its capacity, overloading it and denying "
        "availability to legitimate users."
    ),
}


def narratives_hash() -> str:
    """Return a content hash over the narratives so a change is detectable."""
    payload = json.dumps(CATEGORY_NARRATIVES, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Assignment:
    """One hypothesis placed against the narrative catalogue."""

    text: str
    category: str | None
    similarity: float
    similarities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the assignment as plain data."""
        return {
            "text": self.text,
            "category": self.category,
            "similarity": round(self.similarity, 4),
            "similarities": {k: round(v, 4) for k, v in self.similarities.items()},
        }


class EmbeddingMatcher:
    """Assigns hypothesis text to narrative categories by cosine similarity."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        narratives: dict[str, str] | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.threshold = threshold
        self.narratives = dict(narratives or CATEGORY_NARRATIVES)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._names = sorted(self.narratives)
        self._reference = self._model.encode(
            [self.narratives[name] for name in self._names],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._cache: dict[str, Assignment] = {}

    def provenance(self) -> dict[str, Any]:
        """Return everything needed to reproduce this matcher."""
        import sentence_transformers
        import torch

        return {
            "model_name": self.model_name,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "threshold": self.threshold,
            "narratives_hash": narratives_hash(),
            "categories": self._names,
        }

    def similarities(self, text: str) -> dict[str, float]:
        """Return the cosine similarity of text to every narrative."""
        return self.assign(text).similarities

    def assign(self, text: str) -> Assignment:
        """Place one hypothesis against the narrative catalogue."""
        cleaned = (text or "").strip()
        if not cleaned:
            return Assignment("", None, 0.0, {name: 0.0 for name in self._names})
        cached = self._cache.get(cleaned)
        if cached is not None:
            return cached

        vector = self._model.encode(
            [cleaned], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = {
            name: float(vector @ self._reference[index])
            for index, name in enumerate(self._names)
        }
        best = max(scores, key=scores.get)
        assignment = Assignment(
            text=cleaned,
            category=best if scores[best] >= self.threshold else None,
            similarity=scores[best],
            similarities=scores,
        )
        self._cache[cleaned] = assignment
        return assignment

    def matches_category(
        self, text: str, category: str, competitors: Iterable[str] = ()
    ) -> bool:
        """Return whether text describes the named category.

        The similarity to this category must clear the threshold and must
        exceed the similarity to every competitor narrative the scenario
        declares. Competitors are the scenario's rivals, not the whole
        catalogue: requiring the category to be the closest of all six is a
        stricter rule than the ground truth asks for, and measured against the
        hand labels it costs far more recall than it buys in precision.
        """
        if not category or category not in self.narratives:
            return False
        scores = self.assign(text).similarities
        mine = scores.get(category, 0.0)
        if mine < self.threshold:
            return False
        return all(mine > scores.get(rival, 0.0) for rival in competitors if rival != category)

    def matches_any(self, text: str, categories: Iterable[str]) -> bool:
        """Return whether text describes any of the named categories."""
        return any(self.matches_category(text, category) for category in categories)
