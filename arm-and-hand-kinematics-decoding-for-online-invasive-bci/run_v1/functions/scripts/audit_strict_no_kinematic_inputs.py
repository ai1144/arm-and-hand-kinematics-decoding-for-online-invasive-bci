#!/usr/bin/env python3
"""Audit a strict result folder for accidental kinematic test-input leakage.

The strict suite should contain direct models only.  That means qpos, qvel,
ctrl, and template trajectories are targets or visualization outputs, never
decoder inputs during validation/test.  This script checks the leaderboard
columns written by ``run_strict_direct_output_suite.py`` and fails if a
non-strict row appears. It's basiclly to check and confirm clean test sets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    """Validate that all leaderboard rows are direct and have zero kinematic inputs."""
    parser = argparse.ArgumentParser(description="Audit strict decoder result folder.")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    leaderboard_path = args.result_dir / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard.csv: {leaderboard_path}")

    leaderboard = pd.read_csv(leaderboard_path)
    problems = []
    if "uses_test_kinematic_input" not in leaderboard.columns:
        problems.append("leaderboard.csv has no uses_test_kinematic_input column")
    else:
        used = leaderboard[leaderboard["uses_test_kinematic_input"].astype(str).str.lower().isin(["true", "1"])]
        if not used.empty:
            problems.append(f"{len(used)} rows report uses_test_kinematic_input=True")
    if "kinematic_input_columns" not in leaderboard.columns:
        problems.append("leaderboard.csv has no kinematic_input_columns column")
    else:
        nonzero = pd.to_numeric(leaderboard["kinematic_input_columns"], errors="coerce").fillna(0)
        if (nonzero != 0).any():
            problems.append(f"{int((nonzero != 0).sum())} rows have nonzero kinematic input columns")
    if "model_mode" in leaderboard.columns:
        non_direct = leaderboard[~leaderboard["model_mode"].astype(str).str.startswith("direct")]
        if not non_direct.empty:
            problems.append(f"{len(non_direct)} rows are not direct model modes")
    if leaderboard["experiment_id"].astype(str).str.contains("hybrid|template|ceiling", case=False, regex=True).any():
        problems.append("one or more experiment_id values mention hybrid/template/ceiling")

    audit_path = args.result_dir / "strict_no_kinematic_input_audit.txt"
    if problems:
        audit_path.write_text("FAILED\n" + "\n".join(f"- {p}" for p in problems) + "\n", encoding="utf-8")
        print(audit_path.read_text(), file=sys.stderr)
        return 1
    message = (
        "PASSED\n"
        "All leaderboard rows are direct strict rows, report uses_test_kinematic_input=False, "
        "and report zero kinematic input columns. Random-neural controls in leaderboard.csv "
        "are the main sanity check for whether models use neural structure.\n"
    )
    audit_path.write_text(message, encoding="utf-8")
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
