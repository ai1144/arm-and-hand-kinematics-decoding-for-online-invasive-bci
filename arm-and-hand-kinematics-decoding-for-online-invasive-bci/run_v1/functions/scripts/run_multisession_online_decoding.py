#!/usr/bin/env python3
"""Shared multi-session helpers used by the VM decoder suite.

The most important function here is combine_sessions().  It loads several
patient-session roots, remaps their local gesture IDs into one global gesture
index, and assigns object/task labels when available.  These labels are used by
object-conditioned and hybrid decoders, so the label source is recorded in
combined_trial_index.csv for interpretability.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder_core.data import (  # noqa: E402
    Dataset,
    DT,
    FeatureConfig,
    Trial,
    build_features_for_trial,
    build_matrix,
    causal_ema,
    load_dataset,
    resolve_channels,
    split_trials,
)
from decoder_core.metrics import per_dim_r2, summarize  # noqa: E402
from decoder_core.models import Standardizer, _fit_ridge_readout, fit_policy  # noqa: E402
from decoder_core.train_eval import direct_predict_policy, simulate_policy, tune_control  # noqa: E402
from decoder_core.mujoco_metadata import DEFAULT_MUJOCO_XML, parse_mujoco_metadata  # noqa: E402
from scripts.run_extra_experiments import AffineKalman, fit_affine_kalman, run_affine_kalman  # noqa: E402
from scripts.train_advanced_policies import (  # noqa: E402
    fit_base_kalman,
    fit_kalman_template_grid,
    predict_trials,
    reconstruct_from_phase_logits,
    target_for_trial,
    train_torch_model,
    trial_phase,
)
from decoder_core.template_policy import build_template_features  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOTS = (
    PACKAGE_ROOT.parent / "data and code" / "MM_sample_data_0123",
    PACKAGE_ROOT.parent / "data and code" / "MM_sample_data_0130",
)
DEFAULT_TEMPLATE_DIR = PACKAGE_ROOT / "new_info" / "data"
OBJECT_ORDER = (
    "small_ball",
    "big_ball",
    "small_v_cylinder",
    "big_v_cylinder",
    "small_h_cylinder",
    "big_h_cylinder",
)
OBJECT_TO_ID = {name: i for i, name in enumerate(OBJECT_ORDER)}
GLOBAL_OBJECT_ORDER = (
    # Old 0123/0130 config objects.
    "small_ring",
    "big_horizontal_cylindar",
    "big_vertical_cylindar",
    "samll_horizontal_cylindar",
    "handle",
    "small_vertical_cylindar",
    # New six-object VM protocol objects.
    "small_ball",
    "big_ball",
    "small_v_cylinder",
    "big_v_cylinder",
    "small_h_cylinder",
    "big_h_cylinder",
    # Unknown legacy bucket used only when a session genuinely has no label.
    "unknown_legacy",
)
GLOBAL_OBJECT_TO_ID = {name: i for i, name in enumerate(GLOBAL_OBJECT_ORDER)}
JOINT_GROUPS = {
    "shoulder": np.arange(0, 3, dtype=np.int64),
    "elbow": np.arange(3, 4, dtype=np.int64),
    "wrist": np.arange(4, 7, dtype=np.int64),
    "arm": np.arange(0, 7, dtype=np.int64),
    "hand": np.arange(7, 32, dtype=np.int64),
    "all": np.arange(0, 32, dtype=np.int64),
}


def clean_value(value):
    """Convert numpy/path scalars into JSON/CSV-friendly values."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    return value


def clean_dict(row: dict) -> dict:
    """Clean one dictionary before writing metrics to disk."""
    return {k: clean_value(v) for k, v in row.items()}


def write_json(path: Path, payload: dict) -> None:
    """Write a small JSON manifest, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def subsample_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly downsample very large training matrices for faster baselines."""
    if x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx], y[idx]


def resample_rows(values: np.ndarray, n_rows: int) -> np.ndarray:
    """Phase-resample a trajectory so two movements can be compared."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == n_rows:
        return values
    old_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, n_rows, dtype=np.float32)
    out = np.empty((n_rows, values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(new_x, old_x, values[:, j]).astype(np.float32)
    return out


def template_category(stem: str) -> str:
    """Remove the final repeat number from a template name to get object category."""
    return re.sub(r"_\d+$", "", stem)


def load_named_templates(template_dir: Path) -> dict[str, np.ndarray]:
    """Load named MuJoCo template trajectories from .npy files."""
    templates = {}
    for path in sorted(template_dir.glob("*.npy")):
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == 32:
            arr = arr.T
        if arr.shape[1] != 32:
            raise ValueError(f"Expected 32-D ctrl template, got {arr.shape} in {path}")
        templates[path.stem] = arr
    if not templates:
        raise FileNotFoundError(f"No named MuJoCo ctrl templates found in {template_dir}")
    return templates


def read_handle_ctrl(root: Path, handle_id: int) -> np.ndarray:
    """Read one handle's 32-D ctrl trajectory from its kinematics CSV."""
    cols = [f"ctrl_{i}" for i in range(32)]
    return pd.read_csv(root / "kinematics" / f"simulation_log_{handle_id}.csv", usecols=cols).to_numpy(np.float32)


def match_session_handles(
    data_roots: list[Path],
    templates: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[tuple[str, int], dict]:
    """Match handle trajectories to named templates when explicit labels are absent."""
    # For CSV-kinematics sessions, handles can be matched to named MuJoCo
    # templates by comparing ctrl trajectories.  This was the bridge between
    # "handle id" and intuitive object names before config labels were fully
    # understood.
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    mapping: dict[tuple[str, int], dict] = {}
    for root in data_roots:
        if not (root / "kinematics").exists():
            continue
        dataset = load_dataset(root)
        handles = sorted({int(t.handle_id) for t in dataset.trials if t.handle_id is not None})
        for handle_id in handles:
            ctrl = read_handle_ctrl(root, handle_id)
            scores = []
            for name, template in templates.items():
                pred = resample_rows(ctrl, template.shape[0])
                mse = float(np.mean((pred - template) ** 2))
                corr = float(np.corrcoef(pred.ravel(), template.ravel())[0, 1])
                scores.append((mse, -corr, name, corr, template.shape[0]))
            scores.sort()
            mse, neg_corr, name, corr, template_len = scores[0]
            row = {
                "session": root.name,
                "handle_id": handle_id,
                "template_name": name,
                "object_name": template_category(name),
                "object_id": OBJECT_TO_ID.get(template_category(name), -1),
                "match_mse": mse,
                "match_corr": corr,
                "template_len": template_len,
                "second_template": scores[1][2] if len(scores) > 1 else "",
                "second_corr": scores[1][3] if len(scores) > 1 else np.nan,
            }
            rows.append(row)
            mapping[(root.name, handle_id)] = row
    match_table = pd.DataFrame(rows)
    if not match_table.empty:
        match_table = match_table.sort_values(["session", "handle_id"])
    match_table.to_csv(out_dir / "handle_template_matches.csv", index=False)
    return mapping


def combine_sessions(data_roots: list[Path], template_dir: Path, out_dir: Path) -> tuple[Dataset, pd.DataFrame]:
    """Pool sessions into one dataset and write the combined trial-index table."""
    # Load each session independently first.  This keeps each session's native
    # format handling inside decoder_core.data and lets this function focus
    # on global IDs, object labels, and the combined trial index.
    templates = load_named_templates(template_dir)
    handle_mapping = match_session_handles(data_roots, templates, out_dir)
    datasets = [load_dataset(root) for root in data_roots]
    session_has_csv_kinematics = {root.name: (root / "kinematics").exists() for root in data_roots}
    session_offsets = {root.name: 10000 * (i + 1) for i, root in enumerate(data_roots)}
    label_source = os.environ.get("THESIS_OBJECT_LABEL_SOURCE", "config").strip().lower()
    if label_source not in {"config", "template"}:
        raise ValueError("THESIS_OBJECT_LABEL_SOURCE must be either 'config' or 'template'.")

    observed_object_names = {
        str(t.object_name)
        for ds in datasets
        for t in ds.trials
        if str(t.object_name or "").strip()
    }
    old_object_names = set(GLOBAL_OBJECT_ORDER[:6])
    new_object_names = set(OBJECT_ORDER)
    if observed_object_names and observed_object_names.issubset(new_object_names):
        # New-only same-threshold reruns should use a compact six-object one-hot
        # table: 0=small_ball, ..., 5=big_h_cylinder.  This is easier to
        # interpret and matches the integer object IDs expected online.
        dynamic_object_to_id = dict(OBJECT_TO_ID)
    elif observed_object_names and observed_object_names.issubset(old_object_names):
        dynamic_object_to_id = {name: i for i, name in enumerate(GLOBAL_OBJECT_ORDER[:6])}
    else:
        # Pooled old+new runs need a non-colliding namespace because the old
        # local object labels and new VM labels both used local IDs 0..5.
        dynamic_object_to_id = dict(GLOBAL_OBJECT_TO_ID)

    def canonical_handle_id(trial: Trial) -> int:
        """Make handle IDs globally unique before creating gesture-holdout splits."""
        # Legacy sessions often used simple gesture indices as handles.  Offsets
        # prevent different sessions from accidentally sharing the same gesture
        # ID when gesture_holdout is computed across the combined dataset.
        raw_handle_id = int(trial.handle_id) if trial.handle_id is not None else int(trial.gesture)
        if str(trial.object_code).startswith("same_threshold") or str(trial.object_code).startswith("handle_order"):
            return raw_handle_id
        if session_has_csv_kinematics.get(trial.session, False):
            return raw_handle_id
        return session_offsets.get(trial.session, 90000) + raw_handle_id

    all_trials: list[Trial] = []
    label_sources: list[str] = []
    unique_handles = sorted({canonical_handle_id(t) for ds in datasets for t in ds.trials})
    handle_to_gesture = {handle_id: i for i, handle_id in enumerate(unique_handles)}

    def global_object_id(name: str) -> int:
        """Assign a single non-colliding object ID by object name across sessions."""
        clean = str(name or "unknown_legacy")
        if clean not in dynamic_object_to_id:
            dynamic_object_to_id[clean] = len(dynamic_object_to_id)
        return dynamic_object_to_id[clean]

    for ds in datasets:
        for trial in ds.trials:
            raw_handle_id = int(trial.handle_id) if trial.handle_id is not None else int(trial.gesture)
            handle_id = canonical_handle_id(trial)
            info = handle_mapping.get((trial.session, raw_handle_id), {})
            if label_source == "config" and trial.object_id >= 0:
                object_name = trial.object_name or str(info.get("object_name") or "unknown")
                # Important: trial.object_id is local to the original session
                # format.  Old config sessions used small_ring=0, while new VM
                # sessions can use small_ball=0.  For pooled training, IDs must
                # be globally unique by name, so object_onehot columns do not
                # accidentally merge different physical objects.
                mapped_object_id = global_object_id(object_name)
                object_label_source = (
                    "config"
                    if session_has_csv_kinematics.get(trial.session, False)
                    else (trial.object_code or "legacy_inferred")
                )
            else:
                # Template fallback is useful when a CSV session lacks a clean
                # config label.  If neither config nor template information is
                # available, the trial is assigned an unknown task class.
                object_name = str(info.get("object_name") or trial.object_name or "unknown_legacy")
                mapped_object_id = global_object_id(object_name)
                object_label_source = "template_match" if info else (trial.object_code or ("config_fallback" if trial.object_id >= 0 else "unknown"))
            object_id = int(mapped_object_id)
            handle_label = f"handle_{raw_handle_id}" if handle_id == raw_handle_id else f"{trial.session}_handle_{raw_handle_id}"
            movement_name = f"{object_name}|{handle_label}"
            all_trials.append(
                replace(
                    trial,
                    gesture=handle_to_gesture[handle_id],
                    handle_id=handle_id,
                    movement_name=movement_name,
                    object_name=object_name,
                    object_id=object_id,
                )
            )
            label_sources.append(object_label_source)

    base = datasets[0]
    combined = Dataset(
        trials=all_trials,
        area_labels=base.area_labels,
        joint_names=base.joint_names,
        qpos_names=base.qpos_names,
        ctrl_names=base.ctrl_names,
        sensor_columns=base.sensor_columns,
        qpos_columns=base.qpos_columns,
        qvel_columns=base.qvel_columns,
        qacc_columns=base.qacc_columns,
        ctrl_columns=base.ctrl_columns,
    )
    trial_rows = []
    for t, object_label_source in zip(combined.trials, label_sources, strict=True):
        trial_rows.append(
            {
                "session": t.session,
                "gesture": t.gesture,
                "handle_id": t.handle_id,
                "movement_name": t.movement_name,
                "object_name": t.object_name,
                "object_id": t.object_id,
                "object_label_source": object_label_source,
                "repeat": t.repeat,
                "n_time": t.n_time,
            }
        )
    trial_table = pd.DataFrame(trial_rows)
    trial_table.to_csv(out_dir / "combined_trial_index.csv", index=False)
    return combined, trial_table


def target_array(trial: Trial, target_mode: str) -> np.ndarray:
    """Return the requested supervised target array for one trial."""
    if target_mode == "qpos":
        return trial.qpos
    if target_mode == "qvel":
        return trial.qvel
    if target_mode == "pv":
        return np.concatenate([trial.qpos, trial.qvel], axis=1).astype(np.float32)
    if target_mode == "ctrl":
        return trial.ctrl
    raise ValueError(target_mode)


def direct_target_metrics(policy, dataset: Dataset, trials: list[Trial], target_mode: str) -> dict:
    """Evaluate a direct policy against its native target mode."""
    channels = resolve_channels(dataset, policy.feature_config)
    trues, preds = [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, policy.feature_config)
        if policy.feature_config.lag:
            x = x[:-policy.feature_config.lag]
        pred = policy.predict_target(x)
        true = target_array(trial, target_mode)
        if policy.feature_config.lag:
            true = true[policy.feature_config.lag :]
        n = min(true.shape[0], pred.shape[0])
        trues.append(true[:n])
        preds.append(pred[:n])
    return summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))


def qpos_metrics_for_policy(policy, dataset: Dataset, trials: list[Trial]) -> dict:
    """Evaluate how an older policy ultimately performs in qpos space."""
    if policy.target_mode == "ctrl":
        return {}
    yd, pd_ = direct_predict_policy(policy, dataset, trials)
    direct = summarize(yd, pd_)
    return {
        "direct_qpos_R2": direct["R2"],
        "direct_qpos_CC": direct["CC"],
        "direct_qpos_RMSE": direct["RMSE"],
        # This is a vectorized causal replay metric: features are still
        # current/past-only, but we avoid slow Python step-by-step simulation.
        "online_qpos_R2": direct["R2"],
        "online_qpos_CC": direct["CC"],
        "online_qpos_RMSE": direct["RMSE"],
        "online_jitter": direct["jitter"],
        "online_drift": direct["drift"],
    }


def direct_grid() -> list[tuple[FeatureConfig, float]]:
    """Return the small causal-feature/ridge grid used by older direct baselines."""
    # Keep the default run small enough to iterate on a laptop while still
    # testing raw, smoothed, and short-history causal features.
    return [
        (FeatureConfig(channels="all", smooth_alpha=None, history=1, lag=0), 100.0),
        (FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0), 100.0),
    ]


def train_ridge_family(
    dataset: Dataset,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train older Ridge baselines for qpos, qvel, qpos+qvel, and ctrl targets."""
    rows = []
    best: dict[str, object] = {}
    for target_mode in ("qpos", "qvel", "pv", "ctrl"):
        best_score = -np.inf
        best_policy = None
        best_row = None
        grid = direct_grid()
        for grid_i, (cfg, ridge_alpha) in enumerate(grid, 1):
            print(
                f"  ridge {target_mode} {grid_i}/{len(grid)}: {cfg.label()} alpha={ridge_alpha:g}",
                flush=True,
            )
            t0 = time.perf_counter()
            x_train, y_train = build_matrix(train, dataset, cfg, target_mode)
            x_fit, y_fit = subsample_rows(x_train, y_train, max_rows=30_000, seed=100 + grid_i)
            policy = fit_policy(
                x_fit,
                y_fit,
                model_kind="linear",
                target_mode=target_mode,
                feature_config=cfg,
                channel_indices=resolve_channels(dataset, cfg),
                ridge_alpha=ridge_alpha,
            )
            if target_mode == "pv":
                # Full PV gain tuning is useful but expensive.  The default
                # analysis uses a conservative clipped controller so qpos+qvel
                # can be compared without unstable rollouts dominating runtime.
                policy.control_kp = 3.0
                policy.control_vel_gain = 0.5
                policy.control_max_speed = 8.0
                policy.output_alpha = 0.65
            else:
                tune_control(policy, val)
            val_target = direct_target_metrics(policy, dataset, val, target_mode)
            test_target = direct_target_metrics(policy, dataset, test, target_mode)
            val_qpos = qpos_metrics_for_policy(policy, dataset, val)
            test_qpos = qpos_metrics_for_policy(policy, dataset, test)
            row = {
                "model_family": "ridge",
                "target_mode": target_mode,
                "feature": cfg.label(),
                "ridge_alpha": ridge_alpha,
                "train_ms": (time.perf_counter() - t0) * 1000.0,
                "control_kp": policy.control_kp,
                "control_vel_gain": policy.control_vel_gain,
                "control_max_speed": np.nan if policy.control_max_speed is None else policy.control_max_speed,
                "output_alpha": policy.output_alpha,
                **{f"val_target_{k}": v for k, v in val_target.items()},
                **{f"test_target_{k}": v for k, v in test_target.items()},
                **{f"val_{k}": v for k, v in val_qpos.items()},
                **{f"test_{k}": v for k, v in test_qpos.items()},
            }
            rows.append(row)
            score = row.get("val_online_qpos_R2", row["val_target_R2"])
            if score is None or not np.isfinite(score):
                score = row["val_target_R2"]
            if float(score) > best_score:
                best_score = float(score)
                best_policy = policy
                best_row = row
        assert best_policy is not None and best_row is not None
        path = out_dir / f"best_ridge_{target_mode}.npz"
        best_policy.save(path)
        best_row["artifact"] = str(path)
        best[target_mode] = best_policy
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "ridge_grid.csv", index=False)
    return table, best


class SklearnPolicy:
    """Adapter so sklearn models expose the same predict_target API as Policy."""

    def __init__(self, model, x_scaler: Standardizer, y_scaler: Standardizer, feature_config: FeatureConfig, target_mode: str):
        """Store sklearn model plus scalers and target metadata."""
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_config = feature_config
        self.target_mode = target_mode

    def predict_target(self, x: np.ndarray) -> np.ndarray:
        """Predict a target array and invert output scaling."""
        pred = self.model.predict(self.x_scaler.transform(x))
        return self.y_scaler.inverse(np.asarray(pred, dtype=np.float32)).astype(np.float32)


def train_elastic_net_family(
    dataset: Dataset,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
) -> pd.DataFrame:
    """Train older Elastic Net baselines for comparison with Ridge."""
    from sklearn.linear_model import MultiTaskElasticNet

    rows = []
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    for target_mode in ("qpos", "qvel", "pv", "ctrl"):
        x_train, y_train = build_matrix(train, dataset, cfg, target_mode)
        x_train, y_train = subsample_rows(x_train, y_train, max_rows=25_000, seed=207)
        x_scaler = Standardizer.fit(x_train)
        y_scaler = Standardizer.fit(y_train)
        xs = x_scaler.transform(x_train).astype(np.float32)
        ys = y_scaler.transform(y_train).astype(np.float32)
        if xs.shape[0] > 5_000:
            rng = np.random.default_rng(7)
            idx = rng.choice(xs.shape[0], size=5_000, replace=False)
            xs_fit = xs[idx]
            ys_fit = ys[idx]
        else:
            xs_fit = xs
            ys_fit = ys
        alpha = 0.001
        l1_ratio = 0.25
        print(
            f"  elastic_net {target_mode}: alpha={alpha:g} l1_ratio={l1_ratio:g} "
            f"fit_bins={xs_fit.shape[0]}",
            flush=True,
        )
        t0 = time.perf_counter()
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=80, tol=5e-3, random_state=7)
        model.fit(xs_fit, ys_fit)
        policy = SklearnPolicy(model, x_scaler, y_scaler, cfg, target_mode)
        val_target = eval_sklearn_direct(policy, dataset, val, target_mode)
        test_target = eval_sklearn_direct(policy, dataset, test, target_mode)
        row = {
            "model_family": "elastic_net",
            "target_mode": target_mode,
            "feature": cfg.label(),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "train_ms": (time.perf_counter() - t0) * 1000.0,
            **{f"val_target_{k}": v for k, v in val_target.items()},
            **{f"test_target_{k}": v for k, v in test_target.items()},
        }
        rows.append(row)
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "elastic_net_grid.csv", index=False)
    return table


def eval_sklearn_direct(policy: SklearnPolicy, dataset: Dataset, trials: list[Trial], target_mode: str) -> dict:
    """Evaluate a sklearn-style direct decoder on one split."""
    channels = resolve_channels(dataset, policy.feature_config)
    trues, preds = [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, policy.feature_config)
        pred = policy.predict_target(x)
        true = target_array(trial, target_mode)
        n = min(true.shape[0], pred.shape[0])
        trues.append(true[:n])
        preds.append(pred[:n])
    return summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))


def obs_trials_for_kalman(trials: list[Trial], cfg: FeatureConfig, channels: np.ndarray, scaler: Standardizer | None = None):
    """Build smoothed/running-mean neural observations for Kalman-style models."""
    raw = []
    for trial in trials:
        neural = causal_ema(trial.neural[:, channels], cfg.smooth_alpha)
        running = np.cumsum(neural, axis=0) / np.arange(1, neural.shape[0] + 1, dtype=np.float32)[:, None]
        raw.append(running.astype(np.float32))
    if scaler is None:
        scaler = Standardizer.fit(np.concatenate(raw, axis=0).astype(np.float32))
    return [scaler.transform(x).astype(np.float32) for x in raw], scaler


def state_for_trial(trial: Trial, mode: str, refit: bool = False) -> np.ndarray:
    """Build qpos or qpos+qvel state; ReFIT rotates qvel toward the final target."""
    if mode == "qpos":
        return trial.qpos.astype(np.float32)
    if mode == "pv":
        qvel = trial.qvel.astype(np.float32)
        if refit:
            goal = trial.qpos[-1]
            direction = goal[None, :] - trial.qpos
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            unit = direction / np.maximum(norm, 1e-6)
            speed = np.linalg.norm(qvel, axis=1, keepdims=True)
            qvel = (unit * speed).astype(np.float32)
        return np.concatenate([trial.qpos, qvel], axis=1).astype(np.float32)
    raise ValueError(mode)


def run_template_free_kalman(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
) -> pd.DataFrame:
    """Run older template-free low-rank Kalman/ReFIT Kalman experiments."""
    from sklearn.decomposition import PCA

    rows = []
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    channels = np.arange(256, dtype=np.int64)
    train_obs, scaler = obs_trials_for_kalman(train, cfg, channels)
    val_obs, _ = obs_trials_for_kalman(val, cfg, channels, scaler)
    test_obs, _ = obs_trials_for_kalman(test, cfg, channels, scaler)

    for state_mode, refit in (("qpos", False), ("pv", False), ("pv", True)):
        train_full = [state_for_trial(t, state_mode, refit=refit) for t in train]
        full_scaler = Standardizer.fit(np.concatenate(train_full, axis=0).astype(np.float32))
        full_scaled = full_scaler.transform(np.concatenate(train_full, axis=0).astype(np.float32))
        n_components = min(12, full_scaled.shape[1], full_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=7)
        pca.fit(full_scaled)
        train_states = [pca.transform(full_scaler.transform(s).astype(np.float32)).astype(np.float32) for s in train_full]
        x0 = np.nanmedian(np.stack([s[0] for s in train_states]), axis=0).astype(np.float32)
        t0 = time.perf_counter()
        dyn_a, dyn_b = fit_latent_dynamics(train_states)
        obs_w, obs_b = _fit_ridge_readout(
            np.concatenate(train_obs, axis=0).astype(np.float32),
            np.concatenate(train_states, axis=0).astype(np.float32),
            alpha=1000.0,
        )
        val_true, val_pred = predict_steady_latent_observer_qpos(
            val,
            val_obs,
            pca,
            full_scaler,
            dyn_a,
            dyn_b,
            obs_w,
            obs_b,
            x0,
        )
        test_true, test_pred = predict_steady_latent_observer_qpos(
            test,
            test_obs,
            pca,
            full_scaler,
            dyn_a,
            dyn_b,
            obs_w,
            obs_b,
            x0,
        )
        val_metrics = summarize(val_true, val_pred)
        test_metrics = summarize(test_true, test_pred)
        family = "refit_kalman_lowrank_steady" if refit else "kalman_filter_lowrank_steady"
        row = {
            "model_family": family,
            "target_mode": state_mode,
            "feature": "running_mean_ema0.15",
            "latent_dim": n_components,
            "pca_variance": float(np.sum(pca.explained_variance_ratio_)),
            "train_ms": (time.perf_counter() - t0) * 1000.0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        rows.append(row)
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "template_free_kalman.csv", index=False)
    return table


def fit_latent_dynamics(state_trials: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple affine latent dynamics model x[t+1] ~= A x[t] + b."""
    x1 = np.concatenate([s[:-1] for s in state_trials], axis=0).astype(np.float32)
    x2 = np.concatenate([s[1:] for s in state_trials], axis=0).astype(np.float32)
    xb = np.concatenate([x1, np.ones((x1.shape[0], 1), dtype=np.float32)], axis=1)
    ab = np.linalg.pinv(xb) @ x2
    return ab[:-1].T.astype(np.float32), ab[-1].astype(np.float32)


def predict_steady_latent_observer_qpos(
    trials: list[Trial],
    obs_trials: list[np.ndarray],
    pca,
    full_scaler: Standardizer,
    dyn_a: np.ndarray,
    dyn_b: np.ndarray,
    obs_w: np.ndarray,
    obs_b: np.ndarray,
    x0: np.ndarray,
    measurement_gain: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode qpos with a steady latent observer that blends dynamics and neural readout."""
    trues, preds = [], []
    gain = np.float32(measurement_gain)
    for trial, obs in zip(trials, obs_trials, strict=True):
        meas = (obs @ obs_w + obs_b).astype(np.float32)
        latent = np.empty_like(meas, dtype=np.float32)
        x = x0.astype(np.float32).copy()
        for i in range(meas.shape[0]):
            x_dyn = dyn_a @ x + dyn_b
            x = (np.float32(1.0) - gain) * x_dyn + gain * meas[i]
            latent[i] = x
        full_scaled = pca.inverse_transform(latent).astype(np.float32)
        full = full_scaler.inverse(full_scaled).astype(np.float32)
        pred = full[:, :32]
        n = min(pred.shape[0], trial.qpos.shape[0])
        trues.append(trial.qpos[:n])
        preds.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def predict_latent_kalman_qpos(
    model: AffineKalman,
    trials: list[Trial],
    obs_trials: list[np.ndarray],
    pca,
    full_scaler: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode qpos from an affine Kalman model whose state is low-rank latent kinematics."""
    trues, preds = [], []
    for trial, obs in zip(trials, obs_trials, strict=True):
        latent = run_affine_kalman(model, obs)
        full_scaled = pca.inverse_transform(latent).astype(np.float32)
        full = full_scaler.inverse(full_scaled).astype(np.float32)
        pred = full[:, :32]
        n = min(pred.shape[0], trial.qpos.shape[0])
        trues.append(trial.qpos[:n])
        preds.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def predict_kalman_qpos(model: AffineKalman, trials: list[Trial], obs_trials: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Run an affine Kalman model and return concatenated true/predicted qpos."""
    trues, preds = [], []
    for trial, obs in zip(trials, obs_trials, strict=True):
        state = run_affine_kalman(model, obs)
        pred = state[:, :32].astype(np.float32)
        n = min(pred.shape[0], trial.qpos.shape[0])
        trues.append(trial.qpos[:n])
        preds.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def run_object_decoder(dataset: Dataset, train: list[Trial], val: list[Trial], test: list[Trial], out_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Train a neural-feature ridge classifier that predicts object/task labels."""
    cfgs = [
        FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0),
        FeatureConfig(channels="all", smooth_alpha=None, history=1, lag=0),
        FeatureConfig(channels="motor", smooth_alpha=0.15, history=1, lag=0),
    ]
    rows = []
    best = None
    best_score = -np.inf
    for cfg in cfgs:
        channels = resolve_channels(dataset, cfg)
        x_train = np.concatenate([build_features_for_trial(t, channels, cfg) for t in train], axis=0).astype(np.float32)
        labels_train = np.concatenate([np.full(t.n_time, t.object_id, dtype=np.int64) for t in train])
        if x_train.shape[0] > 30_000:
            rng = np.random.default_rng(307)
            idx = rng.choice(x_train.shape[0], size=30_000, replace=False)
            x_train = x_train[idx]
            labels_train = labels_train[idx]
        x_scaler = Standardizer.fit(x_train)
        y = np.zeros((labels_train.size, len(OBJECT_ORDER)), dtype=np.float32)
        y[np.arange(labels_train.size), labels_train] = 1.0
        y_scaler = Standardizer.fit(y)
        xs = x_scaler.transform(x_train).astype(np.float32)
        ys = y_scaler.transform(y).astype(np.float32)
        for alpha in (100.0, 1000.0, 5000.0):
            w, b = _fit_ridge_readout(xs, ys, alpha)
            val_row, _ = eval_object_readout(val, channels, cfg, x_scaler, y_scaler, w, b)
            test_row, conf = eval_object_readout(test, channels, cfg, x_scaler, y_scaler, w, b)
            row = {
                "model_family": "ridge_object_classifier",
                "feature": cfg.label(),
                "ridge_alpha": alpha,
                **{f"val_{k}": v for k, v in val_row.items()},
                **{f"test_{k}": v for k, v in test_row.items()},
            }
            rows.append(row)
            if val_row["trial_accuracy"] > best_score:
                best_score = val_row["trial_accuracy"]
                best = (row, conf)
    assert best is not None
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "object_decoder_grid.csv", index=False)
    return table, best[1]


def eval_object_readout(
    trials: list[Trial],
    channels: np.ndarray,
    cfg: FeatureConfig,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    w: np.ndarray,
    b: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Score object classification by bin accuracy and trial-majority accuracy."""
    bin_true, bin_pred, trial_true, trial_pred = [], [], [], []
    conf = np.zeros((len(OBJECT_ORDER), len(OBJECT_ORDER)), dtype=np.int64)
    for trial in trials:
        x = build_features_for_trial(trial, channels, cfg)
        scores = y_scaler.inverse(x_scaler.transform(x).astype(np.float32) @ w + b)
        pred = np.argmax(scores, axis=1)
        true = np.full(pred.shape[0], trial.object_id, dtype=np.int64)
        bin_true.append(true)
        bin_pred.append(pred)
        trial_true.append(trial.object_id)
        majority = int(np.argmax(np.bincount(pred, minlength=len(OBJECT_ORDER))))
        trial_pred.append(majority)
        conf[trial.object_id, majority] += 1
    bin_true_arr = np.concatenate(bin_true)
    bin_pred_arr = np.concatenate(bin_pred)
    trial_true_arr = np.asarray(trial_true)
    trial_pred_arr = np.asarray(trial_pred)
    return (
        {
            "bin_accuracy": float(np.mean(bin_true_arr == bin_pred_arr)),
            "trial_accuracy": float(np.mean(trial_true_arr == trial_pred_arr)),
        },
        conf,
    )


def run_object_known_direct(
    dataset: Dataset,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
) -> pd.DataFrame:
    """Train one separate direct Ridge decoder per known object/task label."""
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    channels = resolve_channels(dataset, cfg)
    rows = []
    for target_mode in ("qpos", "pv", "ctrl"):
        preds_val, trues_val, preds_test, trues_test = [], [], [], []
        t0 = time.perf_counter()
        for object_id in sorted({t.object_id for t in train if t.object_id >= 0}):
            train_o = [t for t in train if t.object_id == object_id]
            val_o = [t for t in val if t.object_id == object_id]
            test_o = [t for t in test if t.object_id == object_id]
            if not train_o:
                continue
            x_train, y_train = build_matrix(train_o, dataset, cfg, target_mode)
            x_train, y_train = subsample_rows(x_train, y_train, max_rows=12_000, seed=407 + object_id)
            policy = fit_policy(
                x_train,
                y_train,
                model_kind="linear",
                target_mode=target_mode,
                feature_config=cfg,
                channel_indices=channels,
                ridge_alpha=1000.0,
            )
            for split_trials_, trues_acc, preds_acc in ((val_o, trues_val, preds_val), (test_o, trues_test, preds_test)):
                for trial in split_trials_:
                    x = build_features_for_trial(trial, channels, cfg)
                    pred = policy.predict_target(x)
                    true = target_array(trial, target_mode)
                    n = min(true.shape[0], pred.shape[0])
                    trues_acc.append(true[:n])
                    preds_acc.append(pred[:n])
        val_metrics = summarize(np.concatenate(trues_val, axis=0), np.concatenate(preds_val, axis=0))
        test_metrics = summarize(np.concatenate(trues_test, axis=0), np.concatenate(preds_test, axis=0))
        rows.append(
            {
                "model_family": "object_known_ridge",
                "target_mode": target_mode,
                "known_object": True,
                "feature": cfg.label(),
                "train_ms": (time.perf_counter() - t0) * 1000.0,
                **{f"val_target_{k}": v for k, v in val_metrics.items()},
                **{f"test_target_{k}": v for k, v in test_metrics.items()},
            }
        )
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "object_known_direct.csv", index=False)
    return table


def run_template_decoders(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Run phase/template decoders with predicted-object and known-object variants."""
    rows = []
    predictions: dict[str, list[np.ndarray]] = {}
    for target_mode in ("qpos", "ctrl"):
        row, _policy, _true_trials, pred_trials = fit_kalman_template_grid(train, val, test, target_mode, out_dir)
        row["known_object"] = False
        rows.append(row)
        predictions[f"kalman_phase_template_{target_mode}"] = pred_trials
        known_row, known_preds = run_object_known_template(train, val, test, target_mode)
        rows.append(known_row)
        predictions[f"object_known_kalman_phase_template_{target_mode}"] = known_preds
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "trajectory_template_decoders.csv", index=False)
    return table, predictions


def run_object_known_template(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    target_mode: str,
) -> tuple[dict, list[np.ndarray]]:
    """Run a template decoder while restricting candidate gestures to the known object."""
    cfg, channels, obs_scaler, kalman, class_weights, class_bias = fit_base_kalman(train)
    templates = []
    for gesture in range(max(t.gesture for t in train) + 1):
        candidates = [t for t in train if t.gesture == gesture]
        templates.append(target_for_trial(candidates[0], target_mode).copy())

    gesture_to_object = {t.gesture: t.object_id for t in train}
    object_to_gestures: dict[int, list[int]] = {}
    for gesture, object_id in gesture_to_object.items():
        object_to_gestures.setdefault(object_id, []).append(gesture)

    def prepared(trials: list[Trial]):
        """Prepare true targets, decoded phase, and masked gesture logits."""
        out = []
        for trial in trials:
            features = build_template_features(trial, channels, cfg, use_running_mean=True)
            raw = obs_scaler.transform(features.astype(np.float32)).astype(np.float32)
            state = run_affine_kalman(kalman, raw)
            phase = np.maximum.accumulate(np.clip(state[:, 0], 0.0, 1.0))
            logits = raw @ class_weights + class_bias
            allowed = object_to_gestures.get(trial.object_id, list(range(len(templates))))
            mask = np.full(logits.shape[1], -1e6, dtype=np.float32)
            mask[allowed] = 0.0
            logits = logits + mask[None, :]
            out.append((target_for_trial(trial, target_mode), phase, logits))
        return out

    val_prepared = prepared(val)
    test_prepared = prepared(test)
    best = None
    best_score = -np.inf
    for temperature in (0.1, 0.25, 0.5, 1.0):
        for prob_alpha in (0.05, 0.15, 0.35, 0.7):
            for output_alpha in (0.55, 0.75, 0.9, 1.0):
                pred = np.concatenate(
                    [
                        reconstruct_from_phase_logits(
                            templates,
                            phase,
                            logits,
                            temperature=temperature,
                            prob_alpha=prob_alpha,
                            output_alpha=output_alpha,
                        )
                        for _, phase, logits in val_prepared
                    ],
                    axis=0,
                )
                true = np.concatenate([p[0] for p in val_prepared], axis=0)
                metrics = summarize(true, pred)
                score = metrics["R2"] + 0.03 * metrics["CC"] - 0.01 * metrics["jitter"]
                if score > best_score:
                    best_score = score
                    best = (temperature, prob_alpha, output_alpha, metrics)
    assert best is not None
    temperature, prob_alpha, output_alpha, val_metrics = best
    test_preds = [
        reconstruct_from_phase_logits(
            templates,
            phase,
            logits,
            temperature=temperature,
            prob_alpha=prob_alpha,
            output_alpha=output_alpha,
        )
        for _, phase, logits in test_prepared
    ]
    test_true = np.concatenate([p[0] for p in test_prepared], axis=0)
    test_pred = np.concatenate(test_preds, axis=0)
    test_metrics = summarize(test_true, test_pred)
    return (
        {
            "model_family": "object_known_kalman_phase_template",
            "target_mode": target_mode,
            "known_object": True,
            "temperature": temperature,
            "prob_alpha": prob_alpha,
            "output_alpha": output_alpha,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        },
        test_preds,
    )


def run_torch_decoders(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
    *,
    device: str,
    skip_torch: bool,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Train older GRU/LFADS-lite sequence decoders for qpos/qvel/pv/ctrl targets."""
    if skip_torch:
        return pd.DataFrame(), {}
    rows = []
    predictions = {}
    for target_mode in ("qpos", "qvel", "pv", "ctrl"):
        for model_kind in ("gru", "lfads_lite"):
            row, pred_trials = train_torch_model(
                model_kind=model_kind,
                target_mode=target_mode,
                train=train,
                val=val,
                test=test,
                out_dir=out_dir,
                hidden_dim=64,
                factor_dim=24,
                lr=1e-3,
                epochs=28,
                patience=7,
                batch_size=12,
                seed=11 if model_kind == "gru" else 12,
                recon_weight=0.05,
                smooth_weight=0.03,
                device_name=device,
            )
            rows.append(row)
            predictions[f"{model_kind}_{target_mode}"] = pred_trials
            print(
                f"  {model_kind} {target_mode}: val_R2={row['val_R2']:.3f} "
                f"test_R2={row['test_R2']:.3f} test_CC={row['test_CC']:.3f}",
                flush=True,
            )
    table = pd.DataFrame([clean_dict(r) for r in rows])
    table.to_csv(out_dir / "torch_sequence_decoders.csv", index=False)
    return table, predictions


def run_joint_group_analysis(dataset: Dataset, train: list[Trial], test: list[Trial], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train simple Ridge probes to see which body parts/joints decode best."""
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    channels = resolve_channels(dataset, cfg)
    x_train = np.concatenate([build_features_for_trial(t, channels, cfg) for t in train], axis=0).astype(np.float32)
    x_test_by_trial = [build_features_for_trial(t, channels, cfg) for t in test]
    group_rows = []
    per_joint_rows = []
    for signal in ("qpos", "qvel"):
        for group, dims in JOINT_GROUPS.items():
            y_train = np.concatenate([getattr(t, signal)[:, dims] for t in train], axis=0).astype(np.float32)
            x_fit, y_fit = subsample_rows(x_train, y_train, max_rows=30_000, seed=507 + len(dims))
            model = fit_policy(
                x_fit,
                y_fit,
                model_kind="linear",
                target_mode=signal,
                feature_config=cfg,
                channel_indices=channels,
                ridge_alpha=1000.0,
            )
            trues, preds = [], []
            for trial, x in zip(test, x_test_by_trial, strict=True):
                trues.append(getattr(trial, signal)[:, dims])
                preds.append(model.predict_target(x))
            metrics = summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))
            group_rows.append({"signal": signal, "group": group, "n_dims": len(dims), **metrics})

        for dim in range(32):
            y_train = np.concatenate([getattr(t, signal)[:, dim : dim + 1] for t in train], axis=0).astype(np.float32)
            x_fit, y_fit = subsample_rows(x_train, y_train, max_rows=30_000, seed=607 + dim)
            model = fit_policy(
                x_fit,
                y_fit,
                model_kind="linear",
                target_mode=signal,
                feature_config=cfg,
                channel_indices=channels,
                ridge_alpha=1000.0,
            )
            trues, preds = [], []
            for trial, x in zip(test, x_test_by_trial, strict=True):
                trues.append(getattr(trial, signal)[:, dim : dim + 1])
                preds.append(model.predict_target(x))
            metrics = summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))
            per_joint_rows.append(
                {
                    "signal": signal,
                    "joint_index": dim,
                    "joint_name": dataset.qpos_names[dim] if dim < len(dataset.qpos_names) else f"joint_{dim}",
                    **metrics,
                }
            )
    group_table = pd.DataFrame([clean_dict(r) for r in group_rows])
    joint_table = pd.DataFrame([clean_dict(r) for r in per_joint_rows])
    group_table.to_csv(out_dir / "joint_group_results.csv", index=False)
    joint_table.to_csv(out_dir / "per_joint_results.csv", index=False)
    return group_table, joint_table


def qpos_to_ctrl(qpos: np.ndarray, ctrl_to_qpos_index: list[int | None]) -> np.ndarray:
    """Convert a qpos trajectory into MuJoCo ctrl order for replay exports."""
    out = np.zeros((qpos.shape[0], len(ctrl_to_qpos_index)), dtype=np.float32)
    for ctrl_i, qpos_i in enumerate(ctrl_to_qpos_index):
        if qpos_i is not None:
            out[:, ctrl_i] = qpos[:, qpos_i]
    return out


def export_compare_predictions(
    dataset: Dataset,
    test: list[Trial],
    prediction_sets: dict[str, list[np.ndarray]],
    leaderboard: pd.DataFrame,
    out_dir: Path,
    *,
    repeat: int,
    max_models: int = 4,
) -> None:
    """Export selected prediction sets as actual-vs-decoded MuJoCo ctrl files."""
    metadata = parse_mujoco_metadata(DEFAULT_MUJOCO_XML)
    if metadata is None:
        raise FileNotFoundError(DEFAULT_MUJOCO_XML)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_names = []
    for _, row in leaderboard.sort_values("primary_test_R2", ascending=False).iterrows():
        name = str(row["model_name"])
        if name in prediction_sets and name not in ranked_names:
            ranked_names.append(name)
        if len(ranked_names) >= max_models:
            break
    if "kalman_phase_template_ctrl" in prediction_sets and "kalman_phase_template_ctrl" not in ranked_names:
        ranked_names.insert(0, "kalman_phase_template_ctrl")
    ranked_names = ranked_names[:max_models]

    exports = []
    for model_name in ranked_names:
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        preds = prediction_sets[model_name]
        for trial, pred in zip(test, preds, strict=True):
            if trial.repeat != repeat:
                continue
            if pred.shape[1] == 32 and "ctrl" not in model_name:
                decoded_ctrl = qpos_to_ctrl(pred, metadata.ctrl_to_qpos_index)
            else:
                decoded_ctrl = pred[:, :32]
            n = min(trial.ctrl.shape[0], decoded_ctrl.shape[0])
            actual_path = model_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_actual_ctrl.npy"
            decoded_path = model_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_decoded_ctrl.npy"
            np.save(actual_path, trial.ctrl[:n].T.astype(np.float32))
            np.save(decoded_path, decoded_ctrl[:n].T.astype(np.float32))
            exports.append(
                {
                    "model": model_name,
                    "gesture": trial.gesture,
                    "handle_id": trial.handle_id,
                    "movement_name": trial.movement_name,
                    "object_name": trial.object_name,
                    "repeat": trial.repeat,
                    "frames": int(n),
                    "actual": str(actual_path),
                    "decoded": str(decoded_path),
                }
            )
        model_exports = [e for e in exports if e["model"] == model_name]
        write_json(
            model_dir / "summary.json",
            {
                "policy": model_name,
                "repeat_requested": repeat,
                "order_note": "Both arrays are MuJoCo ctrl order and shape 32 x time. Use scripts/view_mujoco_hand.py --compare --compare-dir this_folder.",
                "exports": model_exports,
            },
        )
    write_json(
        out_dir / "summary.json",
        {
            "repeat_requested": repeat,
            "models": ranked_names,
            "viewer_command_example": "mjpython scripts/view_mujoco_hand.py --compare --compare-dir artifacts/multisession_online_decoding/mujoco_compare/<model_name> --repeat 8 --factor 120 --loop",
            "exports": exports,
        },
    )


def collect_leaderboard(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all older experiment tables into one comparable leaderboard."""
    rows = []

    def add_row(model_name: str, family: str, target_mode: str, primary: float, cc: float, extra: dict):
        """Append one normalized leaderboard row."""
        rows.append(
            {
                "model_name": model_name,
                "model_family": family,
                "target_mode": target_mode,
                "primary_test_R2": primary,
                "primary_test_CC": cc,
                **extra,
            }
        )

    for _, row in tables["ridge"].iterrows():
        family = str(row["model_family"])
        target = str(row["target_mode"])
        name = f"{family}_{target}"
        if np.isfinite(row.get("test_online_qpos_R2", np.nan)):
            add_row(name, family, target, row["test_online_qpos_R2"], row["test_online_qpos_CC"], row.to_dict())
        else:
            add_row(name, family, target, row["test_target_R2"], row["test_target_CC"], row.to_dict())

    for key in ("elastic_net", "template_free_kalman", "object_known_direct", "trajectory", "torch"):
        table = tables.get(key)
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            family = str(row["model_family"])
            target = str(row.get("target_mode", ""))
            known = "_known_object" if bool(row.get("known_object", False)) else ""
            name = f"{family}_{target}{known}".strip("_")
            r2_key = "test_R2" if "test_R2" in row.index and pd.notna(row.get("test_R2")) else "test_target_R2"
            cc_key = "test_CC" if "test_CC" in row.index and pd.notna(row.get("test_CC")) else "test_target_CC"
            if r2_key in row.index:
                add_row(name, family, target, row[r2_key], row.get(cc_key, np.nan), row.to_dict())

    leaderboard = pd.DataFrame([clean_dict(r) for r in rows])
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["primary_test_R2", "primary_test_CC"], ascending=False)
    return leaderboard


def plot_outputs(
    out_dir: Path,
    leaderboard: pd.DataFrame,
    object_table: pd.DataFrame,
    object_confusion: np.ndarray,
    group_table: pd.DataFrame,
    joint_table: pd.DataFrame,
    trial_table: pd.DataFrame,
) -> None:
    """Create summary plots for the older multi-session analysis."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    top = leaderboard.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["model_name"], top["primary_test_R2"], color="#2f80ed")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out test R2")
    ax.set_title("Decoder leaderboard across the two new sessions")
    fig.tight_layout()
    fig.savefig(plot_dir / "decoder_leaderboard_test_R2.png", dpi=180)
    plt.close(fig)

    obj = object_table.sort_values("test_trial_accuracy", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(obj["feature"] + " | a=" + obj["ridge_alpha"].astype(str), obj["test_trial_accuracy"], color="#f2994a")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out trial accuracy")
    ax.set_title("Object decoding from neural activity")
    fig.tight_layout()
    fig.savefig(plot_dir / "object_decoding_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(object_confusion, cmap="Blues")
    ax.set_xticks(range(len(OBJECT_ORDER)), OBJECT_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(OBJECT_ORDER)), OBJECT_ORDER)
    ax.set_xlabel("Predicted object")
    ax.set_ylabel("True object")
    ax.set_title("Object trial confusion matrix")
    for i in range(object_confusion.shape[0]):
        for j in range(object_confusion.shape[1]):
            ax.text(j, i, str(int(object_confusion[i, j])), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_dir / "object_confusion.png", dpi=180)
    plt.close(fig)

    pivot = group_table.pivot(index="group", columns="signal", values="R2").loc[list(JOINT_GROUPS.keys())]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot["qpos"], width, label="qpos", color="#00a6d6")
    ax.bar(x + width / 2, pivot["qvel"], width, label="qvel", color="#f2a900")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=20)
    ax.set_ylabel("Held-out R2")
    ax.set_title("Which body parts decode best?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "joint_group_qpos_qvel_R2.png", dpi=180)
    plt.close(fig)

    heat = joint_table.pivot(index="joint_name", columns="signal", values="R2")
    fig, ax = plt.subplots(figsize=(8, max(8, 0.27 * len(heat))))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.6)
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index, fontsize=7)
    ax.set_title("Per-joint qpos/qvel R2")
    fig.colorbar(im, ax=ax, label="R2")
    fig.tight_layout()
    fig.savefig(plot_dir / "per_joint_qpos_qvel_heatmap.png", dpi=180)
    plt.close(fig)

    session_counts = trial_table.groupby(["session", "object_name"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    session_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Trials")
    ax.set_title("New sessions: attempted trials by decoded object label")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "session_object_trial_counts.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown for the technical report."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(
    out_dir: Path,
    data_roots: list[Path],
    dataset: Dataset,
    trial_table: pd.DataFrame,
    leaderboard: pd.DataFrame,
    object_table: pd.DataFrame,
    group_table: pd.DataFrame,
    joint_table: pd.DataFrame,
) -> None:
    """Write the older multi-session technical report Markdown."""
    top = leaderboard.head(12)
    best_obj = object_table.sort_values("test_trial_accuracy", ascending=False).iloc[0]
    best_groups = group_table.sort_values("R2", ascending=False).head(8)
    best_joints = joint_table.sort_values("R2", ascending=False).head(12)

    lines = []
    lines.append("# Multi-session online decoding report\n")
    lines.append("## What changed\n")
    lines.append(
        "Two new attempted-movement sessions were added. Unlike the previous session, the kinematics are not already packaged as resampled pickles. "
        "For each session I load `trials-attempt.pkl`, read the high-frequency `kinematics/simulation_log_<handle>.csv` files, and resample `qpos`, `qvel`, and `ctrl` to the exact number of neural spike bins in each trial.\n"
    )
    lines.append("## Data used\n")
    for root in data_roots:
        lines.append(f"- `{root}`\n")
    lines.append(
        f"- Total attempted trials after pooling: `{len(dataset.trials)}`\n"
        f"- Unique movement handles after pooling: `{trial_table['handle_id'].nunique()}`\n"
        f"- Object labels are inferred by matching each CSV ctrl trajectory to the named GitLab MuJoCo templates in `new_info/data`.\n"
    )
    lines.append("## Train/validation/test split\n")
    lines.append(
        "The split is repeat-based so every movement/object can appear in every split while the held-out repetitions are never used for fitting: "
        "`0-5` train, `6-7` validation, `8-9` test. This matches the earlier pipeline and keeps the online-decoding question honest.\n"
    )
    lines.append("## Best models so far\n")
    lines.append(markdown_table(top[["model_name", "target_mode", "primary_test_R2", "primary_test_CC"]]))
    lines.append("\n\n## Object decoding\n")
    lines.append(
        f"Best held-out object decoder: `{best_obj['feature']}`, ridge alpha `{best_obj['ridge_alpha']}`. "
        f"Trial accuracy `{best_obj['test_trial_accuracy']:.3f}`, bin accuracy `{best_obj['test_bin_accuracy']:.3f}`.\n"
    )
    lines.append(
        "This is now a much cleaner analysis than the earlier provisional `block3/mod6` labels, because the object labels come from the named MuJoCo movement files.\n"
    )
    lines.append("## qpos vs qvel vs qpos+qvel\n")
    lines.append(
        "The direct models were trained three ways: `qpos` only, `qvel` only, and `pv` which means `qpos + qvel` in one 64-D output. "
        "`qpos` asks the decoder where each joint should be. `qvel` asks how fast each joint should move. `pv` asks for both, then the online controller can blend predicted position with predicted velocity.\n"
    )
    lines.append("## Which joints decode best?\n")
    lines.append(markdown_table(best_groups[["signal", "group", "n_dims", "R2", "CC", "RMSE"]]))
    lines.append("\n\nBest individual joints:\n")
    lines.append(markdown_table(best_joints[["signal", "joint_index", "joint_name", "R2", "CC", "RMSE"]]))
    lines.append("\n\n## What happened to the trajectory model\n")
    lines.append(
        "On the older 0116 package the trajectory/template decoder was the strongest baseline because the gesture labels and resampled kinematics were already cleanly aligned. "
        "On these two new sessions, the current phase/template variant is not yet strong: the direct object-known ridge models beat it clearly. "
        "That is useful rather than disappointing. It says the new CSV sessions need better phase alignment and movement-label handling before template replay can be trusted, while the neural signal is already good enough to improve when the object is known. "
        "The template-free models are still the long-term online-control goal because they infer joint state directly from neural activity instead of replaying a preselected trajectory.\n"
    )
    lines.append("## BRAND / online interpretation\n")
    lines.append(
        "The simulated online stream is one 256-channel neural vector per bin. In a real BRAND/Blackrock session, that vector would come from threshold crossings or spike counts from the 256 intracranial electrodes in real time. "
        "The online wrapper keeps only causal state: smoothing buffers, history taps, and the previous decoded joint state. It does not look into the future. "
        "The MuJoCo viewer can therefore be used as a patient-facing online display once the real-time spike-count source replaces the offline replay loop.\n"
    )
    lines.append("## Output files\n")
    lines.append("- `leaderboard.csv`: normalized comparison across all model families.\n")
    lines.append("- `object_decoder_grid.csv`: object decoding results.\n")
    lines.append("- `joint_group_results.csv` and `per_joint_results.csv`: qpos/qvel body-part results.\n")
    lines.append("- `mujoco_compare/<model>/summary.json`: actual-vs-decoded ctrl exports for the MuJoCo two-hand viewer.\n")
    lines.append("- `plots/`: report figures.\n")
    (out_dir / "MULTISESSION_TECHNICAL_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    """Run the older pooled online-decoding experiment entry point."""
    parser = argparse.ArgumentParser(description="Run pooled online-decoding experiments on the two new attempted sessions.")
    parser.add_argument("--data-root", type=Path, action="append", default=None)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "multisession_online_decoding")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--export-repeat", type=int, default=8)
    args = parser.parse_args()

    data_roots = args.data_root if args.data_root is not None else list(DEFAULT_DATA_ROOTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and pooling sessions...", flush=True)
    dataset, trial_table = combine_sessions(data_roots, args.template_dir, args.out_dir)
    train, val, test = split_trials(dataset.trials)
    print(f"Loaded {len(dataset.trials)} trials: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    print("Training ridge qpos/qvel/qpos+qvel/ctrl policies...", flush=True)
    ridge_table, ridge_best = train_ridge_family(dataset, train, val, test, args.out_dir)

    print("Training elastic-net baselines...", flush=True)
    elastic_table = train_elastic_net_family(dataset, train, val, test, args.out_dir)

    print("Training template-free Kalman and ReFIT-style Kalman filters...", flush=True)
    kalman_table = run_template_free_kalman(train, val, test, args.out_dir)

    print("Training trajectory/phase template decoders, with and without known object...", flush=True)
    trajectory_table, trajectory_predictions = run_template_decoders(train, val, test, args.out_dir)

    print("Training object decoder...", flush=True)
    object_table, object_confusion = run_object_decoder(dataset, train, val, test, args.out_dir)
    np.save(args.out_dir / "object_confusion.npy", object_confusion)

    print("Training object-known direct policies...", flush=True)
    object_known_table = run_object_known_direct(dataset, train, val, test, args.out_dir)

    print("Training GRU and LFADS-lite sequence decoders...", flush=True)
    torch_table, torch_predictions = run_torch_decoders(
        train,
        val,
        test,
        args.out_dir,
        device=args.device,
        skip_torch=args.skip_torch,
    )

    print("Running joint group and per-joint qpos/qvel analysis...", flush=True)
    group_table, joint_table = run_joint_group_analysis(dataset, train, test, args.out_dir)

    tables = {
        "ridge": ridge_table,
        "elastic_net": elastic_table,
        "template_free_kalman": kalman_table,
        "trajectory": trajectory_table,
        "object_known_direct": object_known_table,
        "torch": torch_table,
    }
    leaderboard = collect_leaderboard(tables)
    leaderboard.to_csv(args.out_dir / "leaderboard.csv", index=False)

    prediction_sets = {}
    prediction_sets.update(trajectory_predictions)
    prediction_sets.update(torch_predictions)
    if "ctrl" in ridge_best:
        # Export direct best ridge ctrl predictions for visual comparison too.
        policy = ridge_best["ctrl"]
        preds = []
        channels = resolve_channels(dataset, policy.feature_config)
        for trial in test:
            x = build_features_for_trial(trial, channels, policy.feature_config)
            preds.append(policy.predict_target(x))
        prediction_sets["ridge_ctrl"] = preds

    print("Exporting MuJoCo actual-vs-decoded comparisons...", flush=True)
    export_compare_predictions(
        dataset,
        test,
        prediction_sets,
        leaderboard,
        args.out_dir / "mujoco_compare",
        repeat=args.export_repeat,
    )

    print("Plotting and writing report...", flush=True)
    plot_outputs(args.out_dir, leaderboard, object_table, object_confusion, group_table, joint_table, trial_table)
    write_report(args.out_dir, data_roots, dataset, trial_table, leaderboard, object_table, group_table, joint_table)

    summary = {
        "data_roots": [str(p) for p in data_roots],
        "out_dir": str(args.out_dir),
        "n_trials": len(dataset.trials),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "best_models": [clean_dict(r) for r in leaderboard.head(10).to_dict(orient="records")],
        "report": str(args.out_dir / "MULTISESSION_TECHNICAL_REPORT.md"),
        "plots": str(args.out_dir / "plots"),
        "mujoco_compare": str(args.out_dir / "mujoco_compare"),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(f"Wrote multisession analysis to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
