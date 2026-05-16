"""Utilities to persist every JCE26 execution in a unique results directory.

Each run is stored under:

    RESULTADOS/ejecuciones/YYYY-MM-DD/RUN_NNNN__YYYYMMDD_HHMMSS/

This layout keeps runs ordered by day while also embedding a monotonic execution
number and an exact timestamp in the directory name so no execution overwrites a
previous one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


RUN_DIR_PATTERN = re.compile(r"RUN_(\d{4})__\d{8}_\d{6}$")
RESULTS_ROOT = Path(__file__).resolve().parent / "RESULTADOS"
RUNS_ROOT = RESULTS_ROOT / "ejecuciones"


@dataclass(frozen=True)
class RunDirectories:
    """Filesystem layout used by a single simulation execution."""

    results_root: Path
    runs_root: Path
    date_dir: Path
    run_root: Path
    images: Path
    calculations: Path
    metadata: Path
    run_id: str
    run_number: int
    timestamp: str


def _normalize_for_json(value: Any) -> Any:
    """Recursively convert runtime values into strict JSON-safe objects."""

    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return _normalize_for_json(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting for later inspection."""

    path.write_text(
        json.dumps(_normalize_for_json(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, lines: list[str]) -> None:
    """Write a plain-text log or manifest file from pre-built lines."""

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_run_number(runs_root: Path) -> int:
    """Return the next global execution number across all saved runs."""

    max_run_number = 0
    if runs_root.exists():
        for candidate in runs_root.rglob("RUN_*__*"):
            if not candidate.is_dir():
                continue
            match = RUN_DIR_PATTERN.match(candidate.name)
            if match:
                max_run_number = max(max_run_number, int(match.group(1)))
    return max_run_number + 1


def create_run_directories(now: datetime | None = None) -> RunDirectories:
    """Create and return the directory tree for a new simulation execution."""

    current_time = now or datetime.now().astimezone()
    run_number = _next_run_number(RUNS_ROOT)
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    date_label = current_time.strftime("%Y-%m-%d")
    run_id = f"RUN_{run_number:04d}__{timestamp}"

    date_dir = RUNS_ROOT / date_label
    run_root = date_dir / run_id
    images = run_root / "imagenes"
    calculations = run_root / "calculos"
    metadata = run_root / "metadata"

    for directory in (RESULTS_ROOT, RUNS_ROOT, date_dir, run_root, images, calculations, metadata):
        directory.mkdir(parents=True, exist_ok=True)

    return RunDirectories(
        results_root=RESULTS_ROOT,
        runs_root=RUNS_ROOT,
        date_dir=date_dir,
        run_root=run_root,
        images=images,
        calculations=calculations,
        metadata=metadata,
        run_id=run_id,
        run_number=run_number,
        timestamp=timestamp,
    )
