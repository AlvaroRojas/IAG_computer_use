"""On-disk run artifacts exposed over the API.

The run dir already carries the audit trail written by `diff.write_comparison`
(`comparison/summary.json`, `mismatches.csv`, `report.txt`). This module is the ONLY
place that turns a caller-supplied `run_id` + artifact name into a filesystem path,
so path containment is enforced in exactly one spot:

- the artifact NAME is a whitelist key, never a path segment — the relative part of
  the path is fixed at import time and cannot be influenced by the request;
- `run_dir_for` rejects any `run_id` carrying a path separator and requires the
  resolved directory to sit DIRECTLY under `output_dir`.

Reads never raise: a missing or corrupt artifact yields `None`, so a half-written
run dir can't take the status endpoint down.
"""

from __future__ import annotations

import json
from pathlib import Path

# Public artifact name -> (path relative to the run dir, media type).
ARTIFACTS: dict[str, tuple[str, str]] = {
    "summary.json": ("comparison/summary.json", "application/json"),
    "mismatches.csv": ("comparison/mismatches.csv", "text/csv"),
    "report.txt": ("comparison/report.txt", "text/plain"),
}

SUMMARY_ARTIFACT = "summary.json"


def run_dir_for(output_dir: Path, run_id: str) -> Path | None:
    """The existing run directory for `run_id`, or None when it is missing or the
    id tries to escape `output_dir`."""
    if not run_id or run_id in {".", ".."} or any(s in run_id for s in ("/", "\\")):
        return None
    base = Path(output_dir).resolve()
    candidate = (base / run_id).resolve()
    if candidate.parent != base or not candidate.is_dir():
        return None
    return candidate


def artifact_path(output_dir: Path, run_id: str, name: str) -> Path | None:
    """Path of one artifact, or None when the name is unknown, the run dir is
    invalid, or the file was never produced (a run with no comparison writes none)."""
    entry = ARTIFACTS.get(name)
    run_dir = run_dir_for(output_dir, run_id)
    if entry is None or run_dir is None:
        return None
    path = run_dir / entry[0]
    return path if path.is_file() else None


def media_type(name: str) -> str:
    """Media type for a whitelisted artifact name (KeyError if unknown — callers
    resolve the path first, which already 404s on an unknown name)."""
    return ARTIFACTS[name][1]


def available_artifacts(output_dir: Path, run_id: str) -> list[tuple[str, int]]:
    """`(name, size_bytes)` for every artifact present on disk, in whitelist order.
    Empty while a run is still executing — nothing is written before the comparison
    step, and nothing at all when a side yielded no data (NO_COMPARISON)."""
    found: list[tuple[str, int]] = []
    for name in ARTIFACTS:
        path = artifact_path(output_dir, run_id, name)
        if path is not None:
            found.append((name, path.stat().st_size))
    return found


def read_summary_json(output_dir: Path, run_id: str) -> dict | None:
    """Parsed `comparison/summary.json`, or None when absent/unreadable/not an
    object. Disk is the source of truth here, so it also works for a run whose
    in-memory record was lost."""
    path = artifact_path(output_dir, run_id, SUMMARY_ARTIFACT)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
