#!/usr/bin/env python3
"""Train/evaluate smoothing-oriented decoders from earlier pipeline stages.

This script is kept in the VM bundle because the main suite imports its
smooth_output helper.  It also documents the online-compatible causal smoothing
logic used for visualization cleanup.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder_core.data import FeatureConfig, Trial, build_features_for_trial, build_matrix, resolve_channels, split_trials
from decoder_core.metrics import summarize
from decoder_core.models import Policy, fit_policy
from decoder_core.mujoco_metadata import DEFAULT_MUJOCO_XML, parse_mujoco_metadata
from run_multisession_online_decoding import DEFAULT_DATA_ROOTS, combine_sessions, subsample_rows


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACKAGE_ROOT / "artifacts" / "smooth_decoders"


def clean_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    return value


def clean_dict(row: dict) -> dict:
    return {k: clean_value(v) for k, v in row.items()}


def target_array(trial: Trial, target_mode: str) -> np.ndarray:
    if target_mode == "qpos":
        return trial.qpos
    if target_mode == "ctrl":
        return trial.ctrl
    raise ValueError(target_mode)


def smooth_output(y: np.ndarray, alpha: float, max_delta: np.ndarray | None = None) -> np.ndarray:
    if alpha >= 0.999 and max_delta is None:
        return y.astype(np.float32, copy=True)
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    a = np.float32(alpha)
    for i in range(1, y.shape[0]):
        raw = a * y[i] + (np.float32(1.0) - a) * out[i - 1]
        if max_delta is not None:
            delta = np.clip(raw - out[i - 1], -max_delta, max_delta)
            raw = out[i - 1] + delta
        out[i] = raw
    return out


def fit_global_policy(dataset, train: list[Trial], target_mode: str, cfg: FeatureConfig, max_rows: int, seed: int) -> Policy:
    x_train, y_train = build_matrix(train, dataset, cfg, target_mode)
    x_fit, y_fit = subsample_rows(x_train, y_train, max_rows=max_rows, seed=seed)
    return fit_policy(
        x_fit,
        y_fit,
        model_kind="linear",
        target_mode=target_mode,
        feature_config=cfg,
        channel_indices=resolve_channels(dataset, cfg),
        ridge_alpha=100.0,
    )


def fit_object_policies(dataset, train: list[Trial], target_mode: str, cfg: FeatureConfig, max_rows: int, seed: int):
    policies = {}
    for object_id in sorted({t.object_id for t in train if t.object_id >= 0}):
        train_o = [t for t in train if t.object_id == object_id]
        x_train, y_train = build_matrix(train_o, dataset, cfg, target_mode)
        x_fit, y_fit = subsample_rows(x_train, y_train, max_rows=max_rows, seed=seed + object_id)
        policies[object_id] = fit_policy(
            x_fit,
            y_fit,
            model_kind="linear",
            target_mode=target_mode,
            feature_config=cfg,
            channel_indices=resolve_channels(dataset, cfg),
            ridge_alpha=100.0,
        )
    return policies


def predict_policy_trials(policy: Policy, trials: list[Trial], dataset, target_mode: str, alpha: float, max_delta: np.ndarray | None):
    channels = resolve_channels(dataset, policy.feature_config)
    trues, preds, per_trial = [], [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, policy.feature_config)
        pred = policy.predict_target(x)[:, : target_array(trial, target_mode).shape[1]]
        pred = smooth_output(pred, alpha=alpha, max_delta=max_delta)
        true = target_array(trial, target_mode)
        n = min(true.shape[0], pred.shape[0])
        trues.append(true[:n])
        preds.append(pred[:n])
        per_trial.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0), per_trial


def predict_object_policy_trials(
    policies: dict[int, Policy],
    trials: list[Trial],
    dataset,
    target_mode: str,
    alpha: float,
    max_delta: np.ndarray | None,
):
    trues, preds, per_trial = [], [], []
    for trial in trials:
        policy = policies[trial.object_id]
        channels = resolve_channels(dataset, policy.feature_config)
        x = build_features_for_trial(trial, channels, policy.feature_config)
        pred = policy.predict_target(x)[:, : target_array(trial, target_mode).shape[1]]
        pred = smooth_output(pred, alpha=alpha, max_delta=max_delta)
        true = target_array(trial, target_mode)
        n = min(true.shape[0], pred.shape[0])
        trues.append(true[:n])
        preds.append(pred[:n])
        per_trial.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0), per_trial


def delta_limit_from_train(train: list[Trial], target_mode: str, scale: float = 2.0) -> np.ndarray:
    deltas = []
    for trial in train:
        y = target_array(trial, target_mode)
        if y.shape[0] > 1:
            deltas.append(np.abs(np.diff(y, axis=0)))
    q = np.percentile(np.concatenate(deltas, axis=0), 99.0, axis=0).astype(np.float32)
    return np.maximum(q * np.float32(scale), np.float32(1e-4))


def score_metrics(metrics: dict) -> float:
    return float(metrics["R2"] + 0.03 * metrics["CC"] - 0.025 * metrics["jitter"])


def resample_template(template: np.ndarray, n: int) -> np.ndarray:
    if template.shape[0] == n:
        return template.astype(np.float32, copy=True)
    old = np.linspace(0.0, 1.0, template.shape[0], dtype=np.float32)
    new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.empty((n, template.shape[1]), dtype=np.float32)
    for j in range(template.shape[1]):
        out[:, j] = np.interp(new, old, template[:, j]).astype(np.float32)
    return out


def build_mean_templates(train: list[Trial], target_mode: str, key_name: str) -> dict:
    templates = {}
    for key in sorted({getattr(t, key_name) for t in train}):
        trials = [t for t in train if getattr(t, key_name) == key]
        n = int(np.median([target_array(t, target_mode).shape[0] for t in trials]))
        stack = np.stack([resample_template(target_array(t, target_mode), n) for t in trials], axis=0)
        templates[key] = stack.mean(axis=0).astype(np.float32)
    return templates


def eval_template_upper_bound(train: list[Trial], test: list[Trial], target_mode: str, key_name: str) -> dict:
    templates = build_mean_templates(train, target_mode, key_name)
    trues, preds = [], []
    for trial in test:
        key = getattr(trial, key_name)
        if key not in templates:
            continue
        true = target_array(trial, target_mode)
        pred = resample_template(templates[key], true.shape[0])
        trues.append(true)
        preds.append(pred)
    metrics = summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))
    return {f"test_{k}": v for k, v in metrics.items()}


def qpos_to_ctrl(qpos: np.ndarray, ctrl_to_qpos_index: list[int | None]) -> np.ndarray:
    out = np.zeros((qpos.shape[0], len(ctrl_to_qpos_index)), dtype=np.float32)
    for ctrl_i, qpos_i in enumerate(ctrl_to_qpos_index):
        if qpos_i is not None:
            out[:, ctrl_i] = qpos[:, qpos_i]
    return out


def export_compare(test: list[Trial], preds: list[np.ndarray], target_mode: str, out_dir: Path, repeat: int, model_name: str) -> None:
    metadata = parse_mujoco_metadata(DEFAULT_MUJOCO_XML)
    if metadata is None:
        raise FileNotFoundError(DEFAULT_MUJOCO_XML)
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for trial, pred in zip(test, preds, strict=True):
        if trial.repeat != repeat:
            continue
        decoded_ctrl = pred if target_mode == "ctrl" else qpos_to_ctrl(pred, metadata.ctrl_to_qpos_index)
        n = min(trial.ctrl.shape[0], decoded_ctrl.shape[0])
        actual_path = out_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_actual_ctrl.npy"
        decoded_path = out_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_decoded_ctrl.npy"
        np.save(actual_path, trial.ctrl[:n].T.astype(np.float32))
        np.save(decoded_path, decoded_ctrl[:n].T.astype(np.float32))
        exports.append(
            {
                "gesture": trial.gesture,
                "handle_id": trial.handle_id,
                "object_name": trial.object_name,
                "repeat": trial.repeat,
                "frames": int(n),
                "actual": str(actual_path),
                "decoded": str(decoded_path),
            }
        )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "policy": model_name,
                "repeat_requested": repeat,
                "order_note": "Both arrays are MuJoCo ctrl order and shape 32 x time.",
                "exports": exports,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train jitter-aware smooth decoders for the two new sessions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--template-dir", type=Path, default=PACKAGE_ROOT / "new_info" / "data")
    parser.add_argument("--export-repeat", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset, trial_table = combine_sessions(list(DEFAULT_DATA_ROOTS), args.template_dir, args.out_dir)
    train, val, test = split_trials(dataset.trials)
    cfgs = [
        FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0),
        FeatureConfig(channels="all", smooth_alpha=0.10, history=1, lag=0),
    ]
    output_alphas = [1.0, 0.75, 0.55, 0.35]
    rows = []
    prediction_sets = {}
    best_row = None
    best_preds = None
    best_target = None

    for target_mode in ["qpos", "ctrl"]:
        max_delta = delta_limit_from_train(train, target_mode, scale=1.5)
        for cfg_i, cfg in enumerate(cfgs):
            global_policy = fit_global_policy(dataset, train, target_mode, cfg, max_rows=35_000, seed=700 + cfg_i)
            object_policies = fit_object_policies(dataset, train, target_mode, cfg, max_rows=18_000, seed=800 + cfg_i)
            for known_object, model_name, model in [
                (False, "global_smooth_ridge", global_policy),
                (True, "object_known_smooth_ridge", object_policies),
            ]:
                for alpha in output_alphas:
                    for use_limit in [False, True]:
                        limit = max_delta if use_limit else None
                        if known_object:
                            val_true, val_pred, _ = predict_object_policy_trials(
                                model, val, dataset, target_mode, alpha, limit
                            )
                            test_true, test_pred, test_preds = predict_object_policy_trials(
                                model, test, dataset, target_mode, alpha, limit
                            )
                        else:
                            val_true, val_pred, _ = predict_policy_trials(
                                model, val, dataset, target_mode, alpha, limit
                            )
                            test_true, test_pred, test_preds = predict_policy_trials(
                                model, test, dataset, target_mode, alpha, limit
                            )
                        val_metrics = summarize(val_true, val_pred)
                        test_metrics = summarize(test_true, test_pred)
                        row = {
                            "model_family": model_name,
                            "target_mode": target_mode,
                            "known_object": known_object,
                            "feature": cfg.label(),
                            "output_alpha": alpha,
                            "rate_limited": use_limit,
                            "score": score_metrics(val_metrics),
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                            **{f"test_{k}": v for k, v in test_metrics.items()},
                        }
                        rows.append(row)
                        if best_row is None or row["score"] > best_row["score"]:
                            best_row = row
                            best_preds = test_preds
                            best_target = target_mode

    for target_mode in ["qpos", "ctrl"]:
        for key_name, label in [("handle_id", "handle_known_template_upper_bound"), ("object_id", "object_known_template_upper_bound")]:
            metrics = eval_template_upper_bound(train, test, target_mode, key_name)
            rows.append(
                {
                    "model_family": label,
                    "target_mode": target_mode,
                    "known_object": key_name == "object_id",
                    "feature": "mean_train_template_same_key",
                    "output_alpha": np.nan,
                    "rate_limited": False,
                    "score": metrics["test_R2"],
                    **metrics,
                }
            )

    table = pd.DataFrame([clean_dict(r) for r in rows]).sort_values(["score", "test_R2"], ascending=False)
    table.to_csv(args.out_dir / "smooth_decoder_results.csv", index=False)
    trial_table.to_csv(args.out_dir / "combined_trial_index.csv", index=False)

    if best_row is not None and best_preds is not None and best_target is not None:
        compare_dir = args.out_dir / "mujoco_compare_best_smooth"
        export_compare(test, best_preds, best_target, compare_dir, args.export_repeat, best_row["model_family"])

    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    top = table.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [
        f"{r.model_family}|{r.target_mode}|a={r.output_alpha}|limit={r.rate_limited}"
        for r in top.itertuples(index=False)
    ]
    ax.barh(labels, top["test_R2"], color="#00a6d6")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out test R2")
    ax.set_title("Smooth decoder variants")
    fig.tight_layout()
    fig.savefig(plot_dir / "smooth_decoder_test_R2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = table[table["model_family"].isin(["global_smooth_ridge", "object_known_smooth_ridge"])]
    colors = np.where(scatter["known_object"].astype(bool), "#f2994a", "#2f80ed")
    ax.scatter(scatter["test_jitter"], scatter["test_R2"], c=colors, alpha=0.8)
    ax.set_xlabel("Prediction jitter, lower is smoother")
    ax.set_ylabel("Held-out test R2")
    ax.set_title("R2 vs jitter trade-off")
    fig.tight_layout()
    fig.savefig(plot_dir / "r2_vs_jitter.png", dpi=180)
    plt.close(fig)

    report = [
        "# Smooth decoder experiment",
        "",
        "This experiment keeps the same 0123+0130 pooled split, but adds causal output smoothing and optional rate limiting.",
        "It is meant to answer the practical MuJoCo question: can we make the decoded hand less jittery without destroying R2/CC?",
        "",
        "## Best rows",
        table.head(12).to_markdown(index=False) if False else table.head(12).to_csv(index=False),
        "",
        "## Interpretation",
        "Output smoothing is causal: each frame only uses the current model output and the previous displayed output.",
        "Rate limiting clips frame-to-frame changes to a high percentile of the training movement deltas.",
        "Known-object rows assume an external object cue has already selected the object-specific decoder.",
    ]
    (args.out_dir / "SMOOTH_DECODER_REPORT.md").write_text("\n".join(report))
    summary = {
        "best_row": clean_dict(best_row or {}),
        "results": str(args.out_dir / "smooth_decoder_results.csv"),
        "mujoco_compare": str(args.out_dir / "mujoco_compare_best_smooth"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
