#!/usr/bin/env python3
"""Discover patient attempt-session folders on the VM.

This was the script used to locate the sessions in the VM.  It only decides which folders look
like usable patient sessions and, when --validate is passed, asks the shared
data loader to open them.  The same discovery logic is used by
run_vm_split_pca_suite.py before launching model training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder_core.data import load_dataset


ALL_ATTEMPT_CANDIDATES = (
    "all-attempt.pkl",
    "all_attempt.pkl",
    "all-attempt.pickle",
    "all_attempt.pickle",
)
TRIAL_ATTEMPT_CANDIDATES = (
    "trials-attempt-resampled.pkl",
    "trials_attempt_resampled.pkl",
    "trials-attempt-resample.pkl",
    "trials_attempt_resample.pkl",
    "trials-attempt.pkl",
    "trials_attempt.pkl",
    "trial-attempt.pkl",
    "trial_attempt.pkl",
)
IGNORED_VM_DIR_NAMES = {
    "thresholds defined per block",
    "thresholds_defined_per_block",
    "thresholds-defined-per-block",
}


def _normal_name(name: str) -> str:
    """Normalize file names so dash/underscore variants match the same pattern."""
    return name.lower().replace("_", "-")


def _find_named_pickle(path: Path, candidates: tuple[str, ...]) -> Path | None:
    """Find one expected pickle file while tolerating dash/underscore spelling."""
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    wanted = {_normal_name(name) for name in candidates}
    try:
        files = list(path.iterdir())
    except PermissionError:
        return None
    for child in files:
        if not child.is_file() or child.suffix.lower() not in {".pkl", ".pickle"}:
            continue
        normal = _normal_name(child.name)
        if "flag" in normal or "imagine" in normal or "observe" in normal:
            continue
        if normal in wanted:
            return child
    return None


def is_ignored_vm_path(path: Path) -> bool:
    """Return True for the nested threshold-control folder that is not part of this rerun."""
    return any(part.strip().lower() in IGNORED_VM_DIR_NAMES for part in path.parts)


def is_patient_session(path: Path) -> bool:
    """Return True when a folder has the minimum files needed for attempt trials."""
    # A valid session can be either the legacy pickle-only layout or the newer
    # raw-spike + config.yaml + kinematics CSV layout.  In both cases we require
    # attempt trials because imagine/observe trials are not part of this suite.
    trial_attempt = _find_named_pickle(path, TRIAL_ATTEMPT_CANDIDATES)
    all_attempt = _find_named_pickle(path, ALL_ATTEMPT_CANDIDATES)
    legacy = trial_attempt is not None and all_attempt is not None
    raw = trial_attempt is not None and (path / "config.yaml").exists() and (path / "kinematics").exists()
    return legacy or raw


def session_kind(path: Path) -> str:
    """Label the detected folder layout for the human discovery table."""
    trial_attempt = _find_named_pickle(path, TRIAL_ATTEMPT_CANDIDATES)
    all_attempt = _find_named_pickle(path, ALL_ATTEMPT_CANDIDATES)
    if trial_attempt is not None and all_attempt is not None:
        return "legacy_resampled_0116_style"
    if trial_attempt is not None and (path / "config.yaml").exists() and (path / "kinematics").exists():
        return "raw_spikes_plus_csv_kinematics"
    return "unknown"


def find_patient_sessions(search_roots: list[Path], max_depth: int) -> list[Path]:
    """Recursively discover candidate patient sessions under the given roots."""
    # Breadth-first scan to a bounded depth keeps discovery predictable on a VM
    # share that may contain unrelated folders or OS metadata.
    found: set[Path] = set()
    for root in search_roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        if is_ignored_vm_path(root):
            continue
        if is_patient_session(root):
            found.add(root)
        candidates = [root]
        for _ in range(max_depth):
            next_candidates = []
            for parent in candidates:
                try:
                    for child in parent.iterdir():
                        if not child.is_dir() or child.name.startswith("."):
                            continue
                        if is_ignored_vm_path(child):
                            continue
                        if is_patient_session(child):
                            found.add(child.resolve())
                        next_candidates.append(child)
                except PermissionError:
                    continue
            candidates = next_candidates
    return sorted(found, key=lambda p: p.name)


def summarize_session(path: Path, validate: bool) -> dict:
    """Create one row of the discovery table, optionally loading the data."""
    # The summary table is meant for human sanity checks before training.  If
    # validation is enabled, the object names shown here come from load_dataset.
    row = {
        "path": str(path),
        "name": path.name,
        "kind": session_kind(path),
        "valid_load": "",
        "n_trials": "",
        "n_gestures": "",
        "n_handles": "",
        "n_repeats": "",
        "n_time_min": "",
        "n_time_max": "",
        "objects": "",
        "error": "",
    }
    if not validate:
        return row
    try:
        dataset = load_dataset(path)
        row.update(
            {
                "valid_load": True,
                "n_trials": len(dataset.trials),
                "n_gestures": len({t.gesture for t in dataset.trials}),
                "n_handles": len({t.handle_id for t in dataset.trials if t.handle_id is not None}),
                "n_repeats": len({t.repeat for t in dataset.trials}),
                "n_time_min": min(t.n_time for t in dataset.trials),
                "n_time_max": max(t.n_time for t in dataset.trials),
                "objects": ", ".join(sorted({t.object_name for t in dataset.trials if t.object_name})),
            }
        )
    except Exception as exc:
        row["valid_load"] = False
        row["error"] = repr(exc)
    return row


def main() -> int:
    """Command-line entry point used for manual discovery/debugging."""
    parser = argparse.ArgumentParser(description="Find patient training sessions on a VM.")
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/vm_training/discovery"))
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = find_patient_sessions(args.search_root, args.max_depth)
    rows = [summarize_session(path, args.validate) for path in sessions]
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "discovered_sessions.csv", index=False)

    payload = {
        "search_roots": [str(p.expanduser().resolve()) for p in args.search_root],
        "session_roots": [str(p) for p in sessions],
        "validated": bool(args.validate),
        "n_sessions": len(sessions),
    }
    (args.out_dir / "discovered_sessions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Found {len(sessions)} patient session(s).")
    for path in sessions:
        print(f"  - {path}")
    print(f"Wrote discovery files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
