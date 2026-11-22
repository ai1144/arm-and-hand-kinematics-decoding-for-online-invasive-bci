#!/usr/bin/env python3
"""VM launcher for the patient decoder split/PCA suite.

This file does not train a model itself.  Its job is to:

1. discover valid patient-session folders on the VM, usually under /mnt/vmshare;
2. optionally validate each discovered folder by loading attempt trials;
3. decide whether to run old+new sessions or new sessions only;
4. call the selected training suite with one --data-root per valid session.

The --new-only option is intentionally supported without bundled old sessions.
That allows a small VM upload bundle that contains code only and trains on the
sessions already present on the VM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discover_sessions import find_patient_sessions, summarize_session


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OLD_PATIENT_ROOTS = [
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0116",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0123",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0130",
]
IGNORED_VM_DIR_NAMES = {
    "thresholds defined per block",
    "thresholds_defined_per_block",
    "thresholds-defined-per-block",
}


def is_inside(path: Path, parent: Path) -> bool:
    """Return True when path is located inside parent after resolving symlinks."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def unique_paths(paths: list[Path]) -> list[Path]:
    """Normalize paths and remove duplicates while preserving first-seen order."""
    seen = set()
    out = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def is_ignored_vm_path(path: Path) -> bool:
    """Return True for nested threshold-control folders that must be excluded."""
    return any(part.strip().lower() in IGNORED_VM_DIR_NAMES for part in path.parts)


def session_date_key(path: Path) -> str:
    """Extract a YYYYMMDD date from a VM session/block folder name."""
    match = re.search(r"(20\d{6})", path.name)
    return match.group(1) if match else ""


def find_new_sessions(search_roots: list[Path], explicit_roots: list[Path], max_depth: int) -> list[Path]:
    """Find candidate VM sessions that are not the three bundled old sessions."""
    # Collect explicit roots first, then recursively discover candidate folders.
    # The discovery helper recognizes both legacy all_attempt/trials-attempt
    # folders and the newer config.yaml + kinematics CSV layout.
    found = []
    found.extend([p.expanduser().resolve() for p in explicit_roots])
    for root in search_roots:
        found.extend(find_patient_sessions([root], max_depth=max_depth))
    old_resolved = {p.resolve() for p in OLD_PATIENT_ROOTS if p.exists()}
    old_names = {"MM_sample_data_0116", "MM_sample_data_0123", "MM_sample_data_0130", "session_0123", "session_0130"}
    new = []
    for path in unique_paths(found):
        # These exclusions keep the launcher from accidentally retraining on the
        # old bundled sessions or the older nested threshold-control folder.
        if path in old_resolved:
            continue
        if path.name in old_names:
            continue
        if is_ignored_vm_path(path):
            continue
        if is_inside(path, BUNDLE_ROOT):
            continue
        new.append(path)
    return sorted(new, key=lambda p: str(p))


def shell_join(args: list[str]) -> str:
    """Render a command list exactly enough to paste/debug in a shell log."""
    import shlex

    return " ".join(shlex.quote(str(x)) for x in args)


def main() -> int:
    """Parse launcher options, discover sessions, validate data, and launch training."""
    parser = argparse.ArgumentParser(description="Discover VM sessions and run split/PCA decoder suite.")
    parser.add_argument(
        "--data-search-root",
        type=Path,
        action="append",
        default=[],
        help="Folder to recursively scan for patient sessions, usually /mnt/vmshare on the VM. Can be passed multiple times.",
    )
    parser.add_argument(
        "--new-session-root",
        type=Path,
        action="append",
        default=[],
        help="Explicit patient-session folder to include without recursive discovery. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output result folder. If omitted, a timestamped artifacts/split_pca_suite_* folder is created.",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "quick", "full"],
        default="quick",
        help="Training budget. smoke is a tiny pipeline check; quick is moderate; full is the main thesis-quality run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device passed to PyTorch models: cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"],
        help=(
            "Evaluation split modes to run. repeat_holdout tests later repeats; "
            "gesture_holdout tests unseen handle IDs; session_holdout tests an unseen session date; "
            "block_holdout trains on two blocks and tests on one block."
        ),
    )
    parser.add_argument(
        "--pca-modes",
        nargs="+",
        default=["none", "pca"],
        help="Neural feature modes. none uses causal neural features directly; pca fits PCA on training bins only.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="Number of PCA components when --pca-modes includes pca. Default 64.",
    )
    parser.add_argument(
        "--target-modes",
        nargs="+",
        default=["qpos"],
        choices=["qpos", "qpos_pca", "qvel", "qvel_pca", "pv", "pv_pca", "qpos_qvel", "qpos_qvel_pca"],
        help="Target representations to evaluate. The strict suite supports qpos, qvel, qpos+qvel, and *_pca variants.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Sampling rate used for timing metrics. 100 Hz means one bin is 10 ms.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum recursive search depth under each --data-search-root.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Optional cap on number of model configurations, useful only for debugging.",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Actually load each discovered session before training so bad folders can be skipped early.",
    )
    parser.add_argument(
        "--make-archive",
        action="store_true",
        help="Create a tar.gz archive of the output folder after training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered sessions and the command that would run, but do not train.",
    )
    parser.add_argument(
        "--bidirectional-lfads",
        action="store_true",
        help="Use bidirectional LFADS-lite encoder. This is offline-only because it uses future context.",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=1,
        help="Number of recurrent GRU layers for GRU rows. Use 2 for the BRAND latency/accuracy candidate experiment.",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Train/evaluate only on discovered or explicitly supplied new VM sessions, excluding bundled old sessions.",
    )
    parser.add_argument(
        "--include-handle-ceiling",
        action="store_true",
        help="Forward --include-handle-ceiling to the model suite for repeat-holdout exact-template Ridge.",
    )
    parser.add_argument(
        "--include-qpos-pca-ridge",
        action="store_true",
        help="Forward --include-qpos-pca-ridge to the model suite for reduced-target Ridge diagnostics.",
    )
    parser.add_argument(
        "--qpos-pca-components",
        type=int,
        default=12,
        help="Backward-compatible name for target PCA components.",
    )
    parser.add_argument(
        "--suite-script",
        default="scripts/run_split_pca_model_suite.py",
        help="Training script to run after discovery. Use scripts/run_strict_direct_output_suite.py for the strict direct rerun.",
    )
    parser.add_argument("--export-repeat", type=int, default=8, help="Repeat number exported for MuJoCo replay when present.")
    parser.add_argument("--export-session", default="", help="Optional substring filter for exported MuJoCo replay session names.")
    parser.add_argument("--export-handle-id", type=int, default=None, help="Optional handle ID filter for exported MuJoCo replay trials.")
    parser.add_argument("--export-max-trials", type=int, default=12, help="Maximum trials exported per model for MuJoCo replay.")
    parser.add_argument(
        "--include-date",
        action="append",
        default=[],
        help=(
            "Optional YYYYMMDD date filter for VM sessions. Pass multiple times. "
            "The same-threshold launcher uses this to keep only dates with confirmed handle-to-geom mappings."
        ),
    )
    parser.add_argument(
        "--session-holdout-test-date",
        default="",
        help="Optional YYYYMMDD date to hold out for session_holdout. Empty means latest discovered date.",
    )
    parser.add_argument(
        "--block-holdout-test-block",
        type=int,
        default=3,
        help="Block number held out by block_holdout. Default 3 trains on blocks 1-2 and tests on block 3.",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = (args.out_root or (BUNDLE_ROOT / "artifacts" / f"split_pca_suite_{timestamp}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # old_sessions are the three local sessions shipped inside this pooled VM
    # bundle.  They are required for the original old+new pooled run, but are
    # excluded when --new-only is passed for the follow-up control.
    old_sessions = [p.resolve() for p in OLD_PATIENT_ROOTS if p.exists()]
    if not args.new_only and len(old_sessions) != 3:
        raise SystemExit(f"Expected 3 bundled old sessions, found {len(old_sessions)}: {old_sessions}")
    search_roots = [p.expanduser().resolve() for p in args.data_search_root]
    # candidate_new_sessions are discovered from /mnt/vmshare or explicit
    # --new-session-root folders.  They still need optional load validation.
    candidate_new_sessions = find_new_sessions(search_roots, args.new_session_root, args.max_depth)
    include_dates = {str(x) for x in args.include_date}
    if include_dates:
        candidate_new_sessions = [p for p in candidate_new_sessions if session_date_key(p) in include_dates]

    rows = []
    skipped_new_sessions = []
    # In new-only mode these rows are recorded only if old sessions are present.
    # They are not passed to the model suite.  In pooled mode they are required.
    for path in old_sessions:
        row = summarize_session(path, validate=args.validate_data)
        row["role"] = "bundled_old_session"
        rows.append(row)
        if not args.new_only and args.validate_data and row.get("valid_load") is False:
            raise SystemExit(f"Bundled old session failed validation: {path}\n{row.get('error', '')}")
    # Validate each new VM block.  Bad folders are skipped instead of causing a
    # long run to fail halfway through.  This is why the empty 20260707 folders
    # and malformed single folders do not enter the training command.
    new_sessions = []
    for path in candidate_new_sessions:
        row = summarize_session(path, validate=args.validate_data)
        row["role"] = "new_vm_session"
        rows.append(row)
        if args.validate_data and row.get("valid_load") is False:
            skipped_new_sessions.append((path, row.get("error", "")))
            continue
        new_sessions.append(path)
    # This is the central switch:
    #   pooled run: old bundled sessions + valid new VM blocks
    #   new-only run: valid new VM blocks only
    all_sessions = new_sessions if args.new_only else old_sessions + new_sessions
    if args.new_only and not all_sessions:
        raise SystemExit("No valid new sessions were found, so --new-only cannot run.")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_root / "discovered_sessions.csv", index=False)
    except Exception:
        pass
    discovery = {
        "search_roots": [str(p) for p in search_roots],
        "old_sessions": [str(p) for p in old_sessions],
        "new_sessions": [str(p) for p in new_sessions],
        "skipped_new_sessions": [{"path": str(path), "error": error} for path, error in skipped_new_sessions],
        "all_sessions": [str(p) for p in all_sessions],
        "include_dates": sorted(include_dates),
        "new_only": bool(args.new_only),
        "validated": bool(args.validate_data),
    }
    (out_root / "discovered_sessions.json").write_text(json.dumps(discovery, indent=2), encoding="utf-8")

    print("Old bundled sessions:")
    if old_sessions:
        for path in old_sessions:
            print(f"  - {path}")
    else:
        print("  - none found")
    if args.new_only:
        print("Mode: new-only; old bundled sessions are excluded from training/evaluation.")
    print("New VM sessions:")
    for path in new_sessions:
        print(f"  - {path}")
    if not new_sessions:
        print("  - none found; this will only rerun the bundled old sessions")
    if skipped_new_sessions:
        print("Skipped VM sessions that failed validation:")
        for path, error in skipped_new_sessions:
            print(f"  - {path}: {error}")

    strict_suite = args.suite_script.endswith("run_strict_direct_output_suite.py")

    # Build the exact command for the model suite.  Every valid session becomes
    # one --data-root argument, so the downstream trainer is independent of how
    # discovery happened.
    cmd = [
        sys.executable,
        args.suite_script,
        "--out-dir",
        str(out_root),
        "--profile",
        args.profile,
        "--device",
        args.device,
        "--pca-components",
        str(args.pca_components),
        "--target-modes",
        *args.target_modes,
        "--target-pca-components" if strict_suite else "--qpos-pca-components",
        str(args.qpos_pca_components),
        "--sample-rate-hz",
        str(args.sample_rate_hz),
        "--splits",
        *args.splits,
        "--pca-modes",
        *args.pca_modes,
    ]
    if args.session_holdout_test_date:
        cmd.extend(["--session-holdout-test-date", str(args.session_holdout_test_date)])
    cmd.extend(["--block-holdout-test-block", str(args.block_holdout_test_block)])
    if strict_suite:
        cmd.extend(["--export-repeat", str(args.export_repeat)])
        cmd.extend(["--export-max-trials", str(args.export_max_trials)])
        if args.export_session:
            cmd.extend(["--export-session", str(args.export_session)])
        if args.export_handle_id is not None:
            cmd.extend(["--export-handle-id", str(args.export_handle_id)])
    for path in all_sessions:
        cmd.extend(["--data-root", str(path)])
    if args.max_experiments is not None:
        cmd.extend(["--max-experiments", str(args.max_experiments)])
    if args.bidirectional_lfads:
        cmd.append("--bidirectional-lfads")
    if args.gru_layers != 1:
        cmd.extend(["--gru-layers", str(args.gru_layers)])
    if args.include_handle_ceiling and not strict_suite:
        cmd.append("--include-handle-ceiling")
    if args.include_qpos_pca_ridge and not strict_suite:
        cmd.append("--include-qpos-pca-ridge")

    manifest = {
        **discovery,
        "profile": args.profile,
        "device": args.device,
        "splits": args.splits,
        "pca_modes": args.pca_modes,
        "pca_components": args.pca_components,
        "target_modes": args.target_modes,
        "sample_rate_hz": args.sample_rate_hz,
        "gru_layers": args.gru_layers,
        "include_handle_ceiling": bool(args.include_handle_ceiling),
        "include_qpos_pca_ridge": bool(args.include_qpos_pca_ridge),
        "qpos_pca_components": args.qpos_pca_components,
        "target_pca_components": args.qpos_pca_components,
        "session_holdout_test_date": args.session_holdout_test_date,
        "block_holdout_test_block": args.block_holdout_test_block,
        "suite_script": args.suite_script,
        "export_repeat": args.export_repeat,
        "export_session": args.export_session,
        "export_handle_id": args.export_handle_id,
        "export_max_trials": args.export_max_trials,
        "threshold_condition": "same_thresholds",
        "ignored_vm_folder": "thresholds defined per block",
        "include_dates": sorted(include_dates),
        "command": shell_join(cmd),
        "uses_falcon": False,
        "new_only": bool(args.new_only),
    }
    (out_root / "vm_launcher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCommand:")
    print(shell_join(cmd))

    if not args.dry_run:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BUNDLE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        log_path = out_root / "split_pca_suite.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(shell_join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=BUNDLE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"Training suite failed. Open log: {log_path}")
        if strict_suite:
            audit_cmd = [
                sys.executable,
                "scripts/audit_strict_no_kinematic_inputs.py",
                "--result-dir",
                str(out_root),
            ]
            audit_proc = subprocess.run(audit_cmd, cwd=BUNDLE_ROOT, env=env, text=True)
            if audit_proc.returncode != 0:
                raise SystemExit(f"Strict no-kinematic-input audit failed for {out_root}")
    if args.make_archive:
        archive_base = out_root.with_suffix("")
        archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=out_root)
        print(f"Archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
