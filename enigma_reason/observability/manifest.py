"""
Module: enigma_reason/observability/manifest.py

The provenance record written beside every experiment.

A number in the paper is only defensible if the exact code, configuration,
data and environment that produced it can be named. The manifest carries the
commit of all three repositories rather than one, because a reasoning result
depends on the agent repository for its logic, the machine learning repository
for the sensor that produced its input signals, and the frontend repository
only incidentally but at no cost to record.

Hashes are content hashes of the artefacts actually read, not of the paths
they were read from, so a manifest still identifies its inputs after a file
has been moved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES = 1 << 20
REPOSITORY_NAMES = ("Enigma-AIAgent", "Enigma-ML-Layer", "Enigma-Frontend")


def git_commit(repository: str | Path) -> str:
    """Return the resolved HEAD commit of a repository.

    Args:
        repository: Path to the working tree.

    Returns:
        The full hexadecimal commit, suffixed with "-dirty" when the tree has
        uncommitted changes, or a short diagnostic string when the path is not
        a repository. A missing repository is recorded rather than raised so a
        manifest is still produced on a machine with a partial checkout.
    """
    path = Path(repository)
    if not path.exists():
        return "absent"
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if commit.returncode != 0:
            return "not-a-repository"
        head = commit.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if status.returncode == 0 and status.stdout.strip():
            return f"{head}-dirty"
        return head
    except Exception as exc:
        logger.warning("Git commit unavailable for %s: %s", path, exc)
        return "unavailable"


def file_hash(path: str | Path) -> str:
    """Return the SHA256 of a file, or a diagnostic string when unreadable."""
    target = Path(path)
    if not target.is_file():
        return "absent"
    try:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:
        logger.warning("Hash unavailable for %s: %s", target, exc)
        return "unavailable"


def directory_hash(path: str | Path, patterns: tuple[str, ...] = ("*",)) -> str:
    """Return one hash covering every matching file in a directory tree.

    Files are visited in sorted relative path order and both the path and the
    content contribute, so a rename changes the hash even when no byte of any
    file does.
    """
    root = Path(path)
    if not root.is_dir():
        return "absent"
    try:
        matched: list[Path] = []
        for pattern in patterns:
            matched.extend(p for p in root.rglob(pattern) if p.is_file())
        digest = hashlib.sha256()
        for item in sorted(set(matched), key=lambda p: str(p.relative_to(root))):
            digest.update(str(item.relative_to(root)).encode("utf-8"))
            digest.update(file_hash(item).encode("utf-8"))
        return digest.hexdigest()
    except Exception as exc:
        logger.warning("Directory hash unavailable for %s: %s", root, exc)
        return "unavailable"


def _package_version(name: str) -> str:
    """Return an installed package version without importing heavy modules."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return "absent"
    except Exception as exc:
        logger.warning("Version lookup failed for %s: %s", name, exc)
        return "unavailable"


def environment_fingerprint() -> dict[str, Any]:
    """Describe the machine and interpreter the experiment ran on."""
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "tensorflow_version": _package_version("tensorflow"),
        "langgraph_version": _package_version("langgraph"),
        "langchain_core_version": _package_version("langchain-core"),
        "pydantic_version": _package_version("pydantic"),
    }


def build_run_manifest(
    *,
    experiment: str,
    seed: int,
    config: dict[str, Any],
    started_at: datetime,
    ended_at: datetime | None = None,
    project_root: str | Path | None = None,
    model_artefact: str | Path | None = None,
    dataset: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record for one experiment.

    Args:
        experiment: Short name identifying what was run.
        seed: The seed the experiment was given, recorded even when the
            experiment is deterministic without it.
        config: The resolved configuration, already JSON serialisable.
        started_at: Wall clock start, timezone aware.
        ended_at: Wall clock end. Defaults to now.
        project_root: Directory holding the three repositories. Inferred from
            this file's location when omitted.
        model_artefact: File or directory holding the sensor model whose
            outputs fed this run.
        dataset: File or directory holding the input data.
        extra: Anything experiment specific worth pinning.

    Returns:
        A JSON serialisable manifest.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    finished = ended_at or datetime.now(timezone.utc)

    return {
        "experiment": experiment,
        "seed": seed,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": finished.isoformat(),
        "wall_clock_seconds": round((finished - started_at).total_seconds(), 3),
        "git_commits": {name: git_commit(root / name) for name in REPOSITORY_NAMES},
        "config": config,
        "model_artefact": {
            "path": str(model_artefact) if model_artefact else None,
            "hash": (
                directory_hash(model_artefact)
                if model_artefact and Path(model_artefact).is_dir()
                else file_hash(model_artefact) if model_artefact else "absent"
            ),
        },
        "dataset": {
            "path": str(dataset) if dataset else None,
            "hash": (
                directory_hash(dataset)
                if dataset and Path(dataset).is_dir()
                else file_hash(dataset) if dataset else "absent"
            ),
        },
        "environment": environment_fingerprint(),
        "extra": extra or {},
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a manifest as indented JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
