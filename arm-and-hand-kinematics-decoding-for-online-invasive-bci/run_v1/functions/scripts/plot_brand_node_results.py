#!/usr/bin/env python3
"""Create plots from artificial BRAND-style node tests.

The artificial tests do not measure real online decoding accuracy because they
use fake neural bins, still they are useful for deployment preparation because
they show whether the node loads model artifacts, respects the decode/rest
flag, emits 32-D qpos only during active decoding, and stays under the latency budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def latest_same_threshold_result(root: Path) -> Path:
    """Return the newest same-threshold full result folder under vm_results."""
    candidates = sorted((root / "vm_results").glob("patient_100hz_same_thresholds_strict_full_*"))
    if not candidates:
        candidates = sorted((root / "vm_results").glob("patient_100hz_same_thresholds_full_*"))
    candidates = [p for p in candidates if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No same-threshold full result folder was found.")
    return candidates[-1]


def experiment_id_from_artifact(path_text: str) -> str:
    """Extract the experiment folder name from a saved artifact path."""
    path = Path(path_text)
    try:
        return path.parent.name
    except Exception:
        return path_text


def main() -> int:
    """Read node-test CSV/JSON files and create BRAND-readiness plots."""
    parser = argparse.ArgumentParser(description="Plot artificial BRAND node latency/gating outputs.")
    parser.add_argument("--result-dir", type=Path, default=None, help="VM result folder. Defaults to latest same-threshold full run.")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    result_dir = args.result_dir.resolve() if args.result_dir else latest_same_threshold_result(bundle_root)
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(result_dir.glob("brand_node_artificial_*.json")):
        payload = json.loads(path.read_text())
        artifact = str(payload.get("artifact", ""))
        rows.append(
            {
                "json": str(path),
                "csv": str(path.with_suffix(".csv")),
                "experiment_id": experiment_id_from_artifact(artifact),
                "median_latency_ms": payload.get("median_latency_ms"),
                "p95_latency_ms": payload.get("p95_latency_ms"),
                "target_period_ms": payload.get("target_period_ms", 10.0),
                "n_decoded_bins": payload.get("n_decoded_bins"),
                "meets_100hz_median": payload.get("meets_100hz_median"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No brand_node_artificial_*.json files found in {result_dir}.")

    table = pd.DataFrame(rows)
    leaderboard_path = result_dir / "leaderboard.csv"
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        table = table.merge(
            leaderboard[["experiment_id", "test_R2", "test_CC", "test_RMSE"]],
            on="experiment_id",
            how="left",
        )
    table.to_csv(result_dir / "brand_latency_summary.csv", index=False)

    labels = [x.replace("repeat_holdout__", "").replace("__", "\n") for x in table["experiment_id"]]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(table)), 5))
    ax.bar(x - 0.18, table["median_latency_ms"], width=0.35, label="median", color="#2a9d8f")
    ax.bar(x + 0.18, table["p95_latency_ms"], width=0.35, label="p95", color="#e76f51")
    target = float(np.nanmedian(table["target_period_ms"])) if "target_period_ms" in table else 10.0
    ax.axhline(target, color="black", linestyle="--", linewidth=1.2, label=f"100 Hz budget ({target:.1f} ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Latency per decoded bin (ms)")
    ax.set_title("Artificial BRAND node latency check")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "brand_latency_budget.png", dpi=180)
    plt.close(fig)

    if "test_R2" in table.columns and table["test_R2"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(table["median_latency_ms"], table["test_R2"], color="#4477aa", s=70)
        for _, row in table.iterrows():
            ax.text(row["median_latency_ms"], row["test_R2"], row["experiment_id"].split("__")[2], fontsize=8)
        ax.axvline(target, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Artificial node median latency (ms)")
        ax.set_ylabel("Offline test R2 of same artifact")
        ax.set_title("Deployability view: offline score vs node latency")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "brand_latency_vs_offline_r2.png", dpi=180)
        plt.close(fig)

    first_csv = Path(table.iloc[0]["csv"])
    if first_csv.exists():
        stream = pd.read_csv(first_csv)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        time_s = stream["timestamp_s"].to_numpy(dtype=float)
        decode = stream["decode_enabled"].astype(str).str.lower().isin(["true", "1"]).to_numpy(dtype=float)
        latency = pd.to_numeric(stream["latency_ms"], errors="coerce").to_numpy(dtype=float)
        qpos_cols = [c for c in stream.columns if c.startswith("qpos_")]
        output_present = stream[qpos_cols].notna().any(axis=1).to_numpy(dtype=float) if qpos_cols else decode
        ax1.fill_between(time_s, 0, decode, step="pre", color="#bde0fe", alpha=0.7, label="decode flag")
        ax1.plot(time_s, output_present, color="#2a9d8f", linewidth=1.5, label="qpos output present")
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Flag / output")
        ax2 = ax1.twinx()
        ax2.plot(time_s, latency, color="#e76f51", linewidth=1.0, label="latency")
        ax2.axhline(target, color="black", linestyle="--", linewidth=1.0)
        ax2.set_ylabel("Latency (ms)")
        ax1.set_title("Artificial BRAND decode flag and output timeline")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_dir / "brand_decode_flag_timeline.png", dpi=180)
        plt.close(fig)

    print(f"Wrote BRAND node plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
