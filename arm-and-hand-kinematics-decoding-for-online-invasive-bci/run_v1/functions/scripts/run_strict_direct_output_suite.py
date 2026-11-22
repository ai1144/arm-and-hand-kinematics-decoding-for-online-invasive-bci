#!/usr/bin/env python3
"""Strict neural-only decoder suite for the same-threshold patient VM data.

This script is separate from ``run_split_pca_model_suite.py``:
the older suite included *hybrid/template-assisted* decoders.  Those
hybrid rows did use training-set kinematic templates as input priors.  So for the stricter thesis
rerun we utilized for final results and assesments, this default model grid excludes those template inputs.

The strict suite trains direct decoders only:

- no-label: neural features + elapsed phase -> target;
- object-conditioned: neural features + elapsed phase + object one-hot -> target.

The target can be:

- ``qpos``: 32 joint positions;
- ``qvel``: 32 joint velocities;
- ``pv``: qpos + qvel concatenated, 64 dimensions.

For each target, an optional target PCA variant can be run.  Neural PCA and
target PCA are fit on training data only, then frozen for validation/test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder_core.data import Trial, PROTOCOL_HANDLE_TO_GEOM_ID, VM_BODY_ID_TO_OBJECT
from decoder_core.metrics import mean_corr, per_dim_r2, rmse, summarize
from decoder_core.models import Standardizer, _fit_ridge_readout
from decoder_core.mujoco_metadata import DEFAULT_MUJOCO_XML, parse_mujoco_metadata
from run_multisession_online_decoding import combine_sessions
from run_split_pca_model_suite import (
    N_QPOS,
    clean_dict,
    eval_metric_extras,
    fit_neural_projector,
    kalman_filter_trial,
    latency_ms_per_bin,
    make_split,
    markdown_table,
    neural_features,
    object_onehot,
    one_euro_filter_trials,
    phase_features,
    qpos_to_ctrl,
    save_neural_pca_diagnostics,
    split_pretty_name,
    fit_kalman_matrices,
    corrected_pv_state,
    SequenceNetFactory,
    iter_batches,
    pad_batch,
    masked_delta_mse,
    masked_mse,
    eval_seq_loss,
    session_date_from_name,
    block_index_from_name,
)
from train_smooth_decoders import smooth_output


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OLD_PATIENT_ROOTS = [
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0116",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0123",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0130",
]
DEFAULT_TEMPLATE_DIR = BUNDLE_ROOT / "new_info" / "data"
DEFAULT_TARGET_PCA_COMPONENTS = 12


def canonical_target_mode(mode: str) -> str:
    """Normalize human aliases into the internal target-mode names."""
    mode = str(mode).strip().lower()
    aliases = {
        "qpos+qvel": "pv",
        "qpos_qvel": "pv",
        "qpos+qvel_pca": "pv_pca",
        "qpos_qvel_pca": "pv_pca",
        "position_velocity": "pv",
        "position_velocity_pca": "pv_pca",
    }
    return aliases.get(mode, mode)


def target_base_mode(mode: str) -> str:
    """Return the physical target represented by a direct or PCA target mode."""
    mode = canonical_target_mode(mode)
    if mode.endswith("_pca"):
        mode = mode[:-4]
    if mode not in {"qpos", "qvel", "pv"}:
        raise ValueError(f"Unsupported target mode: {mode}")
    return mode


def target_uses_pca(mode: str) -> bool:
    """Return True when the target is compressed with PCA before prediction."""
    return canonical_target_mode(mode).endswith("_pca")


def target_array(trial: Trial, base_mode: str) -> np.ndarray:
    """Return the supervised target array for one trial."""
    base_mode = target_base_mode(base_mode)
    if base_mode == "qpos":
        return trial.qpos.astype(np.float32)
    if base_mode == "qvel":
        return trial.qvel.astype(np.float32)
    if base_mode == "pv":
        return np.concatenate([trial.qpos, trial.qvel], axis=1).astype(np.float32)
    raise ValueError(base_mode)


def target_dim_names(base_mode: str, qpos_names: list[str] | None = None) -> list[str]:
    """Create interpretable names for target dimensions."""
    base_mode = target_base_mode(base_mode)
    names = list(qpos_names or [f"qpos_{i}" for i in range(N_QPOS)])
    if len(names) != N_QPOS:
        names = [f"qpos_{i}" for i in range(N_QPOS)]
    if base_mode == "qpos":
        return names
    if base_mode == "qvel":
        return [name.replace("qpos", "qvel", 1) if name.startswith("qpos") else f"qvel:{name}" for name in names]
    return names + [f"qvel:{name}" for name in names]


def qpos_from_target_prediction(pred: np.ndarray, trial: Trial, base_mode: str, sample_rate_hz: float) -> np.ndarray:
    """Convert any target prediction into qpos for scoring and MuJoCo replay.

    qpos and pv targets contain qpos directly.  qvel-only models must be
    integrated into positions for visualization.  For offline evaluation, the
    integration starts from the first real qpos sample of the test trial.  That
    starting posture is not a trajectory leak; it represents the robot's known
    current state at attempt onset.  In BRAND, the same role would be played by
    the robot/simulator state stream.
    """
    pred = np.asarray(pred, dtype=np.float32)
    base_mode = target_base_mode(base_mode)
    if base_mode == "qpos":
        return pred[:, :N_QPOS].astype(np.float32)
    if base_mode == "pv":
        return pred[:, :N_QPOS].astype(np.float32)
    if base_mode == "qvel":
        dt = np.float32(1.0 / float(sample_rate_hz))
        qpos0 = trial.qpos[:1].astype(np.float32)
        return (qpos0 + np.cumsum(pred[:, :N_QPOS] * dt, axis=0)).astype(np.float32)
    raise ValueError(base_mode)


def qpos_predictions_from_targets(
    trials: list[Trial],
    preds: list[np.ndarray],
    *,
    base_mode: str,
    sample_rate_hz: float,
) -> list[np.ndarray]:
    """Convert a list of target-space predictions into qpos trajectories."""
    return [
        qpos_from_target_prediction(pred, trial, base_mode, sample_rate_hz)
        for trial, pred in zip(trials, preds, strict=True)
    ]


@dataclass(slots=True)
class TargetCodec:
    """Optional training-only PCA transform for qpos/qvel/qpos+qvel targets."""

    requested_mode: str
    base_mode: str
    full_dim: int
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    explained_variance: float = 1.0

    @property
    def mode(self) -> str:
        """Stable mode name written to CSVs and model artifacts."""
        if self.components is None:
            return self.base_mode
        return f"{self.base_mode}_pca{self.output_dim}"

    @property
    def output_dim(self) -> int:
        """Number of values the model head predicts before reconstruction."""
        if self.components is None:
            return self.full_dim
        return int(self.components.shape[0])

    @property
    def uses_pca(self) -> bool:
        """Whether the target is represented by PCA scores."""
        return self.components is not None

    def encode(self, y: np.ndarray) -> np.ndarray:
        """Convert physical target rows into the trained output space."""
        y = np.asarray(y, dtype=np.float32)
        if self.components is None:
            return y
        if self.mean is None:
            raise ValueError("Target PCA codec is missing its mean.")
        return ((y - self.mean) @ self.components.T).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Convert model outputs back into physical target units."""
        z = np.asarray(z, dtype=np.float32)
        if self.components is None:
            return z
        if self.mean is None:
            raise ValueError("Target PCA codec is missing its mean.")
        return (z @ self.components + self.mean).astype(np.float32)


def fit_target_codec(
    train: list[Trial],
    *,
    requested_mode: str,
    components: int,
    seed: int,
) -> TargetCodec:
    """Fit target PCA on training kinematics only, or return identity codec."""
    requested_mode = canonical_target_mode(requested_mode)
    base_mode = target_base_mode(requested_mode)
    full_dim = N_QPOS if base_mode in {"qpos", "qvel"} else 2 * N_QPOS
    if not target_uses_pca(requested_mode):
        return TargetCodec(requested_mode=requested_mode, base_mode=base_mode, full_dim=full_dim)
    rows = np.concatenate([target_array(t, base_mode) for t in train], axis=0).astype(np.float32)
    n_components = min(int(components), rows.shape[1], rows.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(rows)
    return TargetCodec(
        requested_mode=requested_mode,
        base_mode=base_mode,
        full_dim=full_dim,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def save_target_pca_diagnostics(
    train: list[Trial],
    *,
    split_mode: str,
    base_mode: str,
    out_dir: Path,
    marker_components: int,
    seed: int,
) -> None:
    """Save scree/cumulative-variance plot for kinematic target PCA."""
    base_mode = target_base_mode(base_mode)
    rows = np.concatenate([target_array(t, base_mode) for t in train], axis=0).astype(np.float32)
    n_components = min(rows.shape[1], rows.shape[0] - 1)
    if n_components < 1:
        return
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(rows)
    ratio = pca.explained_variance_ratio_.astype(np.float64)
    cumulative = np.cumsum(ratio)

    diag_dir = out_dir / "pca_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance": cumulative,
        }
    )
    table.to_csv(diag_dir / f"{split_mode}_{base_mode}_target_pca_explained_variance.csv", index=False)

    mark = min(int(marker_components), n_components)
    kept = float(cumulative[mark - 1]) if mark >= 1 else float("nan")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["component"], 100.0 * table["cumulative_explained_variance"], color="#2a9d8f", linewidth=2)
    ax.bar(table["component"], 100.0 * table["explained_variance_ratio"], color="#9ad9ca", alpha=0.55, width=1.0)
    ax.axvline(mark, color="#111111", linestyle="--", linewidth=1)
    ax.text(mark, min(96.0, 100.0 * kept + 3.0), f"{mark} PCs = {100.0 * kept:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n_components)
    ax.set_xlabel("Kinematic principal component")
    ax.set_ylabel("Training target variance explained (%)")
    ax.set_title(f"Target PCA explained variance: {split_pretty_name(split_mode)}, {base_mode}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diag_dir / f"{split_mode}_{base_mode}_target_pca_explained_variance.png", dpi=180)
    plt.close(fig)


def direct_features_strict(
    trial: Trial,
    projector,
    n_objects: int,
    condition_object: bool,
) -> np.ndarray:
    """Build strict direct-model inputs without any qpos/qvel template input."""
    parts = [neural_features(trial, projector), phase_features(trial)]
    if condition_object:
        parts.append(object_onehot(trial, n_objects))
    return np.concatenate(parts, axis=1).astype(np.float32)


def concatenate_direct_xy(
    trials: list[Trial],
    projector,
    n_objects: int,
    condition_object: bool,
    target_codec: TargetCodec,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack all bins from all trials into one X/Y matrix for Ridge."""
    xs = []
    ys = []
    for trial in trials:
        xs.append(direct_features_strict(trial, projector, n_objects, condition_object))
        ys.append(target_codec.encode(target_array(trial, target_codec.base_mode)))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def randomize_neural_trials(train: list[Trial], test: list[Trial], *, seed: int) -> list[Trial]:
    """Create a control test set where neural bins are random but kinematics/labels stay aligned."""
    rng = np.random.default_rng(seed)
    rows = np.concatenate([t.neural for t in train], axis=0).astype(np.float32)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std[std < 1e-6] = 1.0
    out = []
    for trial in test:
        fake = rng.normal(mean, std, size=trial.neural.shape).astype(np.float32)
        out.append(replace(trial, neural=fake))
    return out


def randomize_object_trials(test: list[Trial], n_objects: int, *, seed: int) -> list[Trial]:
    """Create a control test set where object labels are wrong/random."""
    rng = np.random.default_rng(seed)
    out = []
    for trial in test:
        if n_objects <= 1 or int(trial.object_id) < 0:
            out.append(trial)
            continue
        choices = [i for i in range(n_objects) if i != int(trial.object_id)]
        out.append(replace(trial, object_id=int(rng.choice(choices))))
    return out


def aggregate_strict_metrics(
    trials: list[Trial],
    preds_target: list[np.ndarray],
    *,
    base_mode: str,
    sample_rate_hz: float,
) -> dict[str, float]:
    """Compute target-space metrics and qpos-replay metrics for one split."""
    true_target = np.concatenate(
        [target_array(t, base_mode)[: p.shape[0]] for t, p in zip(trials, preds_target, strict=True)],
        axis=0,
    )
    pred_target = np.concatenate(preds_target, axis=0)
    target_metrics = eval_metric_extras(true_target, pred_target, sample_rate_hz=sample_rate_hz)

    qpos_preds = qpos_predictions_from_targets(trials, preds_target, base_mode=base_mode, sample_rate_hz=sample_rate_hz)
    true_qpos = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, qpos_preds, strict=True)], axis=0)
    pred_qpos = np.concatenate(qpos_preds, axis=0)
    qpos_metrics = eval_metric_extras(true_qpos, pred_qpos, sample_rate_hz=sample_rate_hz)

    out = {**target_metrics}
    out.update({f"qpos_{k}": v for k, v in qpos_metrics.items()})
    return out


def qpos_metrics_only(trials: list[Trial], qpos_preds: list[np.ndarray], *, sample_rate_hz: float) -> dict[str, float]:
    """Compute metrics when predictions are already qpos trajectories."""
    true_qpos = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, qpos_preds, strict=True)], axis=0)
    pred_qpos = np.concatenate(qpos_preds, axis=0)
    return eval_metric_extras(true_qpos, pred_qpos, sample_rate_hz=sample_rate_hz)


def add_control_metrics(
    row: dict,
    *,
    train: list[Trial],
    test: list[Trial],
    predict_fn,
    n_objects: int,
    condition_object: bool,
    base_mode: str,
    sample_rate_hz: float,
    seed: int,
) -> None:
    """Add random-input sanity checks to one leaderboard row."""
    random_neural = randomize_neural_trials(train, test, seed=seed)
    random_neural_preds = predict_fn(random_neural)
    metrics = aggregate_strict_metrics(random_neural, random_neural_preds, base_mode=base_mode, sample_rate_hz=sample_rate_hz)
    row.update({f"test_random_neural_{k}": v for k, v in metrics.items()})

    if condition_object:
        random_object = randomize_object_trials(test, n_objects, seed=seed + 1)
        random_object_preds = predict_fn(random_object)
        metrics = aggregate_strict_metrics(random_object, random_object_preds, base_mode=base_mode, sample_rate_hz=sample_rate_hz)
        row.update({f"test_random_object_{k}": v for k, v in metrics.items()})
    else:
        row["test_random_object_R2"] = float("nan")
        row["test_random_object_qpos_R2"] = float("nan")

    # Strict direct models have no kinematic input columns.  This explicit
    # audit column makes the no-test-kinematics constraint visible in the CSV.
    row["uses_test_kinematic_input"] = False
    row["kinematic_input_columns"] = 0


@dataclass(slots=True)
class RidgeDirectDecoder:
    """Simple Ridge model plus preprocessing state."""

    x_scaler: Standardizer
    y_scaler: Standardizer
    weights: np.ndarray
    bias: np.ndarray
    condition_object: bool
    target_codec: TargetCodec

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict physical target values from unscaled direct features."""
        xs = self.x_scaler.transform(x).astype(np.float32)
        z_scaled = xs @ self.weights + self.bias
        z = self.y_scaler.inverse(z_scaled).astype(np.float32)
        return self.target_codec.decode(z)


def train_ridge_direct(
    train: list[Trial],
    projector,
    n_objects: int,
    *,
    condition_object: bool,
    target_codec: TargetCodec,
    alpha: float,
) -> tuple[RidgeDirectDecoder, float]:
    """Fit strict Ridge: neural/phase/(optional object) -> target."""
    x_train, y_train = concatenate_direct_xy(train, projector, n_objects, condition_object, target_codec)
    t0 = time.perf_counter()
    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)
    weights, bias = _fit_ridge_readout(
        x_scaler.transform(x_train).astype(np.float32),
        y_scaler.transform(y_train).astype(np.float32),
        alpha=float(alpha),
    )
    train_ms = (time.perf_counter() - t0) * 1000.0
    return (
        RidgeDirectDecoder(
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            weights=weights,
            bias=bias,
            condition_object=condition_object,
            target_codec=target_codec,
        ),
        train_ms,
    )


def predict_ridge_direct(
    model: RidgeDirectDecoder,
    trials: list[Trial],
    projector,
    n_objects: int,
    *,
    output_alpha: float,
) -> list[np.ndarray]:
    """Predict one strict Ridge target trajectory per trial."""
    preds = []
    for trial in trials:
        x = direct_features_strict(trial, projector, n_objects, model.condition_object)
        pred = model.predict(x)
        preds.append(smooth_output(pred.astype(np.float32), alpha=output_alpha).astype(np.float32))
    return preds


def train_eval_ridge_direct(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector,
    n_objects: int,
    *,
    condition_object: bool,
    target_codec: TargetCodec,
    sample_rate_hz: float,
    seed: int,
) -> tuple[dict, list[np.ndarray], object, float]:
    """Train/evaluate Ridge and select post-output smoothing using validation score."""
    model, train_ms = train_ridge_direct(
        train,
        projector,
        n_objects,
        condition_object=condition_object,
        target_codec=target_codec,
        alpha=100.0,
    )
    best_row = None
    best_preds = None
    best_alpha = 1.0
    for output_alpha in (1.0, 0.75, 0.55, 0.35):
        val_preds = predict_ridge_direct(model, val, projector, n_objects, output_alpha=output_alpha)
        test_preds = predict_ridge_direct(model, test, projector, n_objects, output_alpha=output_alpha)
        row = {
            "model_family": "ridge",
            "model_mode": "direct",
            "condition_object": bool(condition_object),
            "output_alpha": output_alpha,
            "train_ms": train_ms,
            **{f"val_{k}": v for k, v in aggregate_strict_metrics(val, val_preds, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz).items()},
            **{f"test_{k}": v for k, v in aggregate_strict_metrics(test, test_preds, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz).items()},
        }
        row["selection_score"] = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row
            best_preds = test_preds
            best_alpha = output_alpha
    assert best_row is not None and best_preds is not None

    def predict_fn(split_trials: list[Trial]) -> list[np.ndarray]:
        return predict_ridge_direct(model, split_trials, projector, n_objects, output_alpha=best_alpha)

    best_row["latency_ms_per_bin_offline"] = latency_ms_per_bin(
        lambda: predict_fn(test),
        total_bins=sum(t.n_time for t in test),
    )
    add_control_metrics(
        best_row,
        train=train,
        test=test,
        predict_fn=predict_fn,
        n_objects=n_objects,
        condition_object=condition_object,
        base_mode=target_codec.base_mode,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    return best_row, best_preds, predict_fn, train_ms


def sequence_items_direct(
    trials: list[Trial],
    projector,
    n_objects: int,
    *,
    condition_object: bool,
    target_codec: TargetCodec,
    x_scaler: Standardizer | None,
    y_scaler: Standardizer | None,
) -> tuple[list[dict], Standardizer, Standardizer]:
    """Prepare one variable-length supervised sequence per trial."""
    raw = []
    xs = []
    ys = []
    for trial in trials:
        x = direct_features_strict(trial, projector, n_objects, condition_object)
        y = target_codec.encode(target_array(trial, target_codec.base_mode))
        raw.append({"trial": trial, "x": x, "y": y})
        xs.append(x)
        ys.append(y)
    if x_scaler is None:
        x_scaler = Standardizer.fit(np.concatenate(xs, axis=0))
    if y_scaler is None:
        y_scaler = Standardizer.fit(np.concatenate(ys, axis=0))
    for item in raw:
        item["x"] = x_scaler.transform(item["x"]).astype(np.float32)
        item["y"] = y_scaler.transform(item["y"]).astype(np.float32)
    return raw, x_scaler, y_scaler


def predict_sequence_direct(
    model,
    items: list[dict],
    y_scaler: Standardizer,
    target_codec: TargetCodec,
    torch,
    device,
    *,
    output_alpha: float,
) -> list[np.ndarray]:
    """Decode target trajectories from a trained GRU/LFADS/Transformer."""
    model.eval()
    preds = []
    with torch.no_grad():
        for item in items:
            xb = torch.from_numpy(item["x"][None]).to(device)
            mask = torch.ones((1, item["x"].shape[0]), dtype=torch.bool, device=device)
            out = model(xb, mask)["y"][0].detach().cpu().numpy().astype(np.float32)
            target = target_codec.decode(y_scaler.inverse(out).astype(np.float32))
            preds.append(smooth_output(target, alpha=output_alpha).astype(np.float32))
    return preds


def train_sequence_direct(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector,
    n_objects: int,
    *,
    model_family: str,
    condition_object: bool,
    target_codec: TargetCodec,
    device_name: str,
    profile: str,
    seed: int,
    bidirectional_lfads: bool,
    gru_layers: int,
    sample_rate_hz: float,
    out_dir: Path,
) -> tuple[dict, list[np.ndarray], object, pd.DataFrame]:
    """Train/evaluate one strict GRU/LFADS-lite/Transformer configuration."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device(device_name)
    train_items, x_scaler, y_scaler = sequence_items_direct(
        train,
        projector,
        n_objects,
        condition_object=condition_object,
        target_codec=target_codec,
        x_scaler=None,
        y_scaler=None,
    )
    val_items, _, _ = sequence_items_direct(
        val,
        projector,
        n_objects,
        condition_object=condition_object,
        target_codec=target_codec,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    test_items, _, _ = sequence_items_direct(
        test,
        projector,
        n_objects,
        condition_object=condition_object,
        target_codec=target_codec,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    if profile == "smoke":
        hidden_dim, factor_dim, epochs, patience, batch_size = 40, 16, 2, 2, 24
    elif profile == "quick":
        hidden_dim, factor_dim, epochs, patience, batch_size = 56, 24, 16, 5, 32
    else:
        hidden_dim, factor_dim, epochs, patience, batch_size = 96, 32, 70, 12, 32

    model = SequenceNetFactory.make(
        torch,
        nn,
        model_family=model_family,
        input_dim=train_items[0]["x"].shape[1],
        output_dim=train_items[0]["y"].shape[1],
        hidden_dim=hidden_dim,
        factor_dim=factor_dim,
        bidirectional_lfads=bidirectional_lfads,
        gru_layers=gru_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history = []
    recon_weight = 0.025 if model_family == "lfads_lite" else 0.0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in iter_batches(train_items, batch_size, rng, shuffle=True):
            xb, yb, mask = pad_batch(batch, torch, device)
            out = model(xb, mask)
            loss = masked_mse(out["y"], yb, mask) + 0.05 * masked_delta_mse(out["y"], yb, mask)
            if out["recon"] is not None:
                loss = loss + recon_weight * masked_mse(out["recon"], xb, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        val_loss = eval_seq_loss(model, val_items, torch, device, batch_size, recon_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"{model_family}/direct/obj={condition_object}/{target_codec.mode} "
            f"epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_ms = (time.perf_counter() - t0) * 1000.0

    best_row = None
    best_preds = None
    best_alpha = 1.0
    for output_alpha in (1.0, 0.75, 0.55, 0.35):
        val_preds = predict_sequence_direct(
            model,
            val_items,
            y_scaler,
            target_codec,
            torch,
            device,
            output_alpha=output_alpha,
        )
        test_preds = predict_sequence_direct(
            model,
            test_items,
            y_scaler,
            target_codec,
            torch,
            device,
            output_alpha=output_alpha,
        )
        row = {
            "model_family": model_family,
            "model_mode": "direct",
            "condition_object": bool(condition_object),
            "target_mode": target_codec.mode,
            "target_base_mode": target_codec.base_mode,
            "target_pca_components": target_codec.output_dim if target_codec.uses_pca else np.nan,
            "target_pca_explained_variance": target_codec.explained_variance if target_codec.uses_pca else np.nan,
            "output_alpha": output_alpha,
            "hidden_dim": hidden_dim,
            "factor_dim": factor_dim if model_family == "lfads_lite" else np.nan,
            "bidirectional": bool(bidirectional_lfads and model_family == "lfads_lite"),
            "gru_layers": int(gru_layers) if model_family == "gru" else np.nan,
            "best_epoch": best_epoch,
            "train_ms": train_ms,
            **{f"val_{k}": v for k, v in aggregate_strict_metrics(val, val_preds, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz).items()},
            **{f"test_{k}": v for k, v in aggregate_strict_metrics(test, test_preds, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz).items()},
        }
        row["selection_score"] = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row
            best_preds = test_preds
            best_alpha = output_alpha
    assert best_row is not None and best_preds is not None

    artifact = out_dir / f"{model_family}_direct_{'obj' if condition_object else 'nolabel'}_{target_codec.mode}.pt"
    torch.save(
        {
            "model_family": model_family,
            "model_mode": "direct",
            "condition_object": bool(condition_object),
            "target_mode": target_codec.mode,
            "target_base_mode": target_codec.base_mode,
            "target_mean": target_codec.mean,
            "target_components": target_codec.components,
            "target_explained_variance": target_codec.explained_variance,
            "projector": {
                "mode": projector.mode,
                "mean": projector.mean,
                "components": projector.components,
                "explained_variance": projector.explained_variance,
            },
            "x_mean": x_scaler.mean,
            "x_scale": x_scaler.scale,
            "y_mean": y_scaler.mean,
            "y_scale": y_scaler.scale,
            "hidden_dim": hidden_dim,
            "factor_dim": factor_dim,
            "gru_layers": int(gru_layers),
            "output_dim": target_codec.output_dim,
            "sample_rate_hz": sample_rate_hz,
            "state_dict": best_state,
        },
        artifact,
    )

    def predict_fn(split_trials: list[Trial]) -> list[np.ndarray]:
        split_items, _, _ = sequence_items_direct(
            split_trials,
            projector,
            n_objects,
            condition_object=condition_object,
            target_codec=target_codec,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        return predict_sequence_direct(
            model,
            split_items,
            y_scaler,
            target_codec,
            torch,
            device,
            output_alpha=best_alpha,
        )

    best_row["latency_ms_per_bin_offline"] = latency_ms_per_bin(
        lambda: predict_fn(test),
        total_bins=sum(t.n_time for t in test),
    )
    add_control_metrics(
        best_row,
        train=train,
        test=test,
        predict_fn=predict_fn,
        n_objects=n_objects,
        condition_object=condition_object,
        base_mode=target_codec.base_mode,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    best_row["artifact"] = str(artifact)
    history_df = pd.DataFrame(history)
    history_df["model_family"] = model_family
    history_df["model_mode"] = "direct"
    history_df["condition_object"] = bool(condition_object)
    history_df["target_mode"] = target_codec.mode
    return best_row, best_preds, predict_fn, history_df


def train_eval_kalman_direct(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector,
    *,
    model_family: str,
    base_mode: str,
    sample_rate_hz: float,
    seed: int,
    n_objects: int,
) -> tuple[dict, list[np.ndarray], object]:
    """Train/evaluate Kalman or ReFIT Kalman for qpos/qvel/pv outputs."""
    refit = model_family == "refit_kalman"
    states = [corrected_pv_state(t, refit=refit) for t in train]
    obs = [neural_features(t, projector) for t in train]
    t0 = time.perf_counter()
    params = fit_kalman_matrices(states, obs, ridge=1e-2)
    train_ms = (time.perf_counter() - t0) * 1000.0

    def z_to_target(z: np.ndarray) -> np.ndarray:
        if base_mode == "qpos":
            return z[:, :N_QPOS].astype(np.float32)
        if base_mode == "qvel":
            return z[:, N_QPOS : 2 * N_QPOS].astype(np.float32)
        if base_mode == "pv":
            return z[:, : 2 * N_QPOS].astype(np.float32)
        raise ValueError(base_mode)

    def predict_fn(split_trials: list[Trial]) -> list[np.ndarray]:
        out = []
        for trial in split_trials:
            z = kalman_filter_trial(params, neural_features(trial, projector))
            out.append(z_to_target(z))
        return out

    val_preds = predict_fn(val)
    test_preds = predict_fn(test)
    row = {
        "model_family": model_family,
        "model_mode": "direct_pv_state",
        "condition_object": False,
        "target_mode": base_mode,
        "target_base_mode": base_mode,
        "target_pca_components": np.nan,
        "target_pca_explained_variance": np.nan,
        "output_alpha": 1.0,
        "train_ms": train_ms,
        "latency_ms_per_bin_offline": latency_ms_per_bin(
            lambda: predict_fn(test),
            total_bins=sum(t.n_time for t in test),
        ),
        **{f"val_{k}": v for k, v in aggregate_strict_metrics(val, val_preds, base_mode=base_mode, sample_rate_hz=sample_rate_hz).items()},
        **{f"test_{k}": v for k, v in aggregate_strict_metrics(test, test_preds, base_mode=base_mode, sample_rate_hz=sample_rate_hz).items()},
    }
    row["selection_score"] = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
    add_control_metrics(
        row,
        train=train,
        test=test,
        predict_fn=predict_fn,
        n_objects=n_objects,
        condition_object=False,
        base_mode=base_mode,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    return row, test_preds, predict_fn


def export_mujoco_compare_selected(
    test: list[Trial],
    qpos_preds: list[np.ndarray],
    out_dir: Path,
    model_label: str,
    *,
    max_trials: int,
    export_repeat: int | None,
    export_session: str,
    export_handle_id: int | None,
) -> None:
    """Save actual-vs-decoded ctrl arrays for a consistent subset of trials."""
    metadata = parse_mujoco_metadata(DEFAULT_MUJOCO_XML)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for trial, pred in zip(test, qpos_preds, strict=True):
        if export_repeat is not None and int(trial.repeat) != int(export_repeat):
            continue
        if export_session and export_session not in str(trial.session):
            continue
        if export_handle_id is not None and int(trial.handle_id or -999999) != int(export_handle_id):
            continue
        chosen.append((trial, pred))
    if not chosen:
        chosen = list(zip(test, qpos_preds, strict=True))
    exports = []
    for trial, pred in chosen[:max_trials]:
        decoded_ctrl = qpos_to_ctrl(pred.astype(np.float32), metadata.ctrl_to_qpos_index)
        n = min(decoded_ctrl.shape[0], trial.ctrl.shape[0])
        session_token = str(trial.session).replace("/", "_")
        stem = f"{session_token}__gesture_{trial.gesture:03d}_repeat_{trial.repeat:02d}"
        actual_path = out_dir / f"{stem}_actual_ctrl.npy"
        decoded_path = out_dir / f"{stem}_decoded_ctrl.npy"
        np.save(actual_path, trial.ctrl[:n].T.astype(np.float32))
        np.save(decoded_path, decoded_ctrl[:n].T.astype(np.float32))
        exports.append(
            {
                "session": trial.session,
                "gesture": int(trial.gesture),
                "handle_id": None if trial.handle_id is None else int(trial.handle_id),
                "object_name": trial.object_name,
                "object_id": int(trial.object_id),
                "repeat": int(trial.repeat),
                "frames": int(n),
                "actual": str(actual_path),
                "decoded": str(decoded_path),
            }
        )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": model_label,
                "decoded_output": "model target converted to qpos, then mapped into MuJoCo ctrl order for replay",
                "selection": {
                    "export_repeat": export_repeat,
                    "export_session_contains": export_session,
                    "export_handle_id": export_handle_id,
                    "max_trials": max_trials,
                },
                "exports": exports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def append_trial_metrics(
    out_dir: Path,
    *,
    experiment_id: str,
    row: dict,
    trials: list[Trial],
    preds_target: list[np.ndarray],
    qpos_preds: list[np.ndarray],
    sample_rate_hz: float,
    output_variant: str,
) -> None:
    """Append per-trial target and qpos metrics for mean/std plots."""
    rows = []
    for trial, pred_target, pred_qpos in zip(trials, preds_target, qpos_preds, strict=True):
        nt = min(target_array(trial, row["target_base_mode"]).shape[0], pred_target.shape[0])
        nq = min(trial.qpos.shape[0], pred_qpos.shape[0])
        target_metrics = eval_metric_extras(
            target_array(trial, row["target_base_mode"])[:nt],
            pred_target[:nt],
            sample_rate_hz=sample_rate_hz,
        )
        qpos_metrics = eval_metric_extras(trial.qpos[:nq], pred_qpos[:nq], sample_rate_hz=sample_rate_hz)
        rows.append(
            {
                "experiment_id": experiment_id,
                "output_variant": output_variant,
                "model_family": row.get("model_family"),
                "model_mode": row.get("model_mode"),
                "condition_object": row.get("condition_object"),
                "policy_label": row.get("policy_label"),
                "pca_condition": row.get("pca_condition"),
                "split_mode": row.get("split_mode"),
                "split_label": split_pretty_name(str(row.get("split_mode"))),
                "target_mode": row.get("target_mode"),
                "target_base_mode": row.get("target_base_mode"),
                "session": trial.session,
                "session_date": session_date_from_name(trial.session),
                "block": block_index_from_name(trial.session),
                "gesture": int(trial.gesture),
                "handle_id": None if trial.handle_id is None else int(trial.handle_id),
                "object_name": trial.object_name,
                "object_id": int(trial.object_id),
                "repeat": int(trial.repeat),
                "n_time": int(nq),
                **{f"target_{k}": v for k, v in target_metrics.items()},
                **{f"qpos_{k}": v for k, v in qpos_metrics.items()},
            }
        )
    if rows:
        path = out_dir / "trial_metrics.csv"
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def append_per_dimension_metrics(
    out_dir: Path,
    *,
    experiment_id: str,
    row: dict,
    trials: list[Trial],
    preds_target: list[np.ndarray],
    qpos_preds: list[np.ndarray],
    qpos_names: list[str],
) -> None:
    """Save R2/CC/RMSE per qpos dimension and per trained target dimension."""
    rows = []
    true_qpos = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, qpos_preds, strict=True)], axis=0)
    pred_qpos = np.concatenate(qpos_preds, axis=0)
    qpos_r2 = per_dim_r2(true_qpos, pred_qpos)
    for dim, name in enumerate(target_dim_names("qpos", qpos_names)):
        rows.append(
            {
                "experiment_id": experiment_id,
                "space": "qpos_replay",
                "dimension": dim,
                "dimension_name": name,
                "R2": float(qpos_r2[dim]),
                "CC": float(mean_corr(true_qpos[:, dim : dim + 1], pred_qpos[:, dim : dim + 1])),
                "RMSE": float(rmse(true_qpos[:, dim : dim + 1], pred_qpos[:, dim : dim + 1])),
                "model_family": row.get("model_family"),
                "policy_label": row.get("policy_label"),
                "split_mode": row.get("split_mode"),
                "target_mode": row.get("target_mode"),
                "pca_condition": row.get("pca_condition"),
            }
        )
    true_target = np.concatenate(
        [target_array(t, row["target_base_mode"])[: p.shape[0]] for t, p in zip(trials, preds_target, strict=True)],
        axis=0,
    )
    pred_target = np.concatenate(preds_target, axis=0)
    target_r2 = per_dim_r2(true_target, pred_target)
    for dim, name in enumerate(target_dim_names(row["target_base_mode"], qpos_names)):
        rows.append(
            {
                "experiment_id": experiment_id,
                "space": "trained_target",
                "dimension": dim,
                "dimension_name": name,
                "R2": float(target_r2[dim]),
                "CC": float(mean_corr(true_target[:, dim : dim + 1], pred_target[:, dim : dim + 1])),
                "RMSE": float(rmse(true_target[:, dim : dim + 1], pred_target[:, dim : dim + 1])),
                "model_family": row.get("model_family"),
                "policy_label": row.get("policy_label"),
                "split_mode": row.get("split_mode"),
                "target_mode": row.get("target_mode"),
                "pca_condition": row.get("pca_condition"),
            }
        )
    if rows:
        path = out_dir / "per_dimension_metrics.csv"
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def policy_label(row: dict) -> str:
    """Short policy label for figures."""
    if bool(row.get("condition_object", False)):
        return "object"
    return "no label"


def pca_condition_label(row: dict) -> str:
    """Short PCA-condition label for figures."""
    neural = "N-PCA" if str(row.get("pca_mode", "none")).startswith("pca") else "no N-PCA"
    target = "Y-PCA" if bool(row.get("target_pca", False)) else "no Y-PCA"
    return f"{neural}, {target}"


def jitter_x(text: str, scale: float = 0.18) -> float:
    """Deterministic tiny horizontal jitter for scatter plots."""
    return ((hash(text) % 1000) / 999.0 - 0.5) * scale


def write_group_metric_summaries(out_dir: Path) -> pd.DataFrame:
    """Create mean/std tables by session, block, object, and model condition."""
    path = out_dir / "trial_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    trial_metrics = pd.read_csv(path)
    raw = trial_metrics[trial_metrics["output_variant"] == "raw"].copy()
    if raw.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in [
            "target_R2",
            "target_CC",
            "target_RMSE",
            "target_procrustes_corr",
            "target_jitter",
            "target_onset_delay_s",
            "target_offset_delay_s",
            "qpos_R2",
            "qpos_CC",
            "qpos_RMSE",
            "qpos_procrustes_corr",
            "qpos_jitter",
            "qpos_onset_delay_s",
            "qpos_offset_delay_s",
        ]
        if col in raw.columns
    ]
    grouping_sets = {
        "session_date": ["experiment_id", "session_date"],
        "session_block": ["experiment_id", "session"],
        "object": ["experiment_id", "object_name"],
        "split_model_policy": ["split_mode", "target_base_mode", "model_family", "policy_label", "pca_condition"],
    }
    frames = []
    for grouping, group_cols in grouping_sets.items():
        grouped = raw.groupby(group_cols, dropna=False)
        for metric in metric_cols:
            stat = grouped[metric].agg(["mean", "std", "count"]).reset_index()
            stat["metric"] = metric
            stat["grouping"] = grouping
            stat = stat.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n_trials"})
            frames.append(stat)
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(out_dir / "metric_mean_std_by_group.csv", index=False)
    return summary


def plot_clean_suite(out_dir: Path, leaderboard: pd.DataFrame, history: pd.DataFrame, trial_table: pd.DataFrame) -> None:
    """Make reviewer-friendly plots with short labels, legends, and std bars."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if leaderboard.empty:
        return
    df = leaderboard.copy()
    df["split_label"] = df["split_mode"].map(split_pretty_name)

    model_colors = {
        "ridge": "#4477aa",
        "kalman": "#cc6677",
        "refit_kalman": "#aa3377",
        "gru": "#228833",
        "lfads_lite": "#ee7733",
        "transformer": "#8856a7",
    }
    policy_markers = {"no label": "o", "object": "s"}
    split_order = [s for s in ["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"] if s in set(df["split_mode"])]
    target_order = [t for t in ["qpos", "qvel", "pv"] if t in set(df["target_base_mode"])]

    # Main strict plot: target R2 by split and output target.
    fig, axes = plt.subplots(1, max(1, len(target_order)), figsize=(5.5 * max(1, len(target_order)), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, target_order, strict=False):
        part = df[df["target_base_mode"] == target]
        x_positions = {split: i for i, split in enumerate(split_order)}
        for _, row in part.iterrows():
            key = f"{row['model_family']}{row['policy_label']}{row['pca_condition']}{row['target_mode']}"
            ax.scatter(
                x_positions[row["split_mode"]] + jitter_x(key),
                row["test_R2"],
                color=model_colors.get(row["model_family"], "#555555"),
                marker=policy_markers.get(row["policy_label"], "o"),
                s=58 if not row["target_pca"] else 36,
                alpha=0.82 if not row["target_pca"] else 0.5,
                edgecolor="black",
                linewidth=0.3,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([x_positions[s] for s in split_order])
        ax.set_xticklabels([split_pretty_name(s).replace(" ", "\n") for s in split_order])
        ax.set_title(target.replace("pv", "qpos+qvel"))
        ax.set_xlabel("split")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Test target R2")
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in model_colors.items()
        if k in set(df["model_family"])
    ]
    policy_handles = [
        plt.Line2D([0], [0], marker=m, linestyle="", color="black", label=label, markersize=7)
        for label, m in policy_markers.items()
        if label in set(df["policy_label"])
    ]
    axes[-1].legend(handles=model_handles + policy_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Strict direct decoders: target R2")
    fig.tight_layout()
    fig.savefig(plot_dir / "strict_direct_target_r2_by_split.png", dpi=180)
    plt.close(fig)

    # qpos replay quality, important when the model was trained on qvel or pv.
    fig, ax = plt.subplots(figsize=(11, 5))
    best = df.sort_values("test_qpos_R2", ascending=False).head(18).iloc[::-1]
    labels = [
        f"{r.model_family} | {r.policy_label} | {r.target_base_mode} | {r.pca_condition} | {split_pretty_name(r.split_mode)}"
        for r in best.itertuples(index=False)
    ]
    ax.barh(labels, best["test_qpos_R2"], color=[model_colors.get(x, "#555555") for x in best["model_family"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Test qpos R2 after reconstruction for MuJoCo")
    ax.set_title("Best qpos-replay rows")
    fig.tight_layout()
    fig.savefig(plot_dir / "strict_best_qpos_replay_r2.png", dpi=180)
    plt.close(fig)

    # Mean/std across test trials for top rows.
    summary = write_group_metric_summaries(out_dir)
    trial_path = out_dir / "trial_metrics.csv"
    if trial_path.exists():
        tm = pd.read_csv(trial_path)
        raw = tm[tm["output_variant"] == "raw"].copy()
        if not raw.empty:
            top_ids = df.sort_values("test_qpos_R2", ascending=False).head(14)["experiment_id"].tolist()
            show = raw[raw["experiment_id"].isin(top_ids)]
            stats = show.groupby("experiment_id")["qpos_R2"].agg(["mean", "std"]).reindex(top_ids).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(stats))))
            y = np.arange(len(stats))
            ax.errorbar(stats["mean"], y, xerr=stats["std"].fillna(0.0), fmt="o", color="#2a9d8f", capsize=3)
            labels = []
            for exp in stats.index:
                row = df[df["experiment_id"] == exp].iloc[0]
                labels.append(f"{row.model_family} | {row.policy_label} | {row.target_base_mode} | {row.pca_condition} | {split_pretty_name(row.split_mode)}")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Per-trial qpos R2, mean +/- std")
            ax.set_title("Top strict rows with trial variability")
            ax.grid(alpha=0.25, axis="x")
            fig.tight_layout()
            fig.savefig(plot_dir / "strict_top_qpos_r2_mean_std.png", dpi=180)
            plt.close(fig)

            # Object performance for the single best qpos replay model per split.
            rows = []
            for split in split_order:
                split_rows = df[df["split_mode"] == split].sort_values("test_qpos_R2", ascending=False)
                if split_rows.empty:
                    continue
                exp = split_rows.iloc[0]["experiment_id"]
                part = raw[raw["experiment_id"] == exp]
                stats_obj = part.groupby("object_name")["qpos_R2"].agg(["mean", "std", "count"]).reset_index()
                stats_obj["split_mode"] = split
                stats_obj["experiment_id"] = exp
                rows.append(stats_obj)
            if rows:
                object_df = pd.concat(rows, ignore_index=True)
                object_df.to_csv(out_dir / "best_model_object_qpos_r2_mean_std.csv", index=False)
                fig, axes = plt.subplots(1, len(split_order), figsize=(5.3 * len(split_order), 5), sharey=True)
                axes = np.atleast_1d(axes)
                for ax, split in zip(axes, split_order, strict=False):
                    part = object_df[object_df["split_mode"] == split].sort_values("mean")
                    ax.barh(part["object_name"], part["mean"], xerr=part["std"].fillna(0.0), color="#00a6d6", alpha=0.85)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_title(split_pretty_name(split))
                    ax.set_xlabel("qpos R2 mean +/- std")
                    ax.grid(alpha=0.25, axis="x")
                fig.suptitle("Best strict decoder: per-object qpos R2")
                fig.tight_layout()
                fig.savefig(plot_dir / "strict_best_model_object_qpos_r2.png", dpi=180)
                plt.close(fig)

            # Session variance across sections/dates.
            session_stats = (
                raw[raw["experiment_id"].isin(top_ids)]
                .groupby(["experiment_id", "session_date"])["qpos_R2"]
                .agg(["mean", "std"])
                .reset_index()
            )
            if not session_stats.empty:
                session_stats.to_csv(out_dir / "top_model_session_qpos_r2_mean_std.csv", index=False)
                fig, ax = plt.subplots(figsize=(11, 5))
                for i, exp in enumerate(top_ids[:8]):
                    part = session_stats[session_stats["experiment_id"] == exp]
                    if part.empty:
                        continue
                    ax.errorbar(
                        part["session_date"],
                        part["mean"],
                        yerr=part["std"].fillna(0.0),
                        marker="o",
                        linewidth=1.2,
                        capsize=3,
                        label=f"row {i + 1}",
                    )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Per-trial qpos R2, mean +/- std")
                ax.set_xlabel("session date")
                ax.set_title("Session-to-session variance for top strict rows")
                ax.legend(ncol=4, fontsize=7)
                ax.grid(alpha=0.25, axis="y")
                fig.tight_layout()
                fig.savefig(plot_dir / "strict_session_variance_top_rows.png", dpi=180)
                plt.close(fig)

    # Per-joint qpos R2 for the best strict qpos-replay model.
    per_dim_path = out_dir / "per_dimension_metrics.csv"
    if per_dim_path.exists():
        per_dim = pd.read_csv(per_dim_path)
        qpos_dim = per_dim[per_dim["space"] == "qpos_replay"].copy()
        if not qpos_dim.empty:
            best_exp = df.sort_values("test_qpos_R2", ascending=False).iloc[0]["experiment_id"]
            part = qpos_dim[qpos_dim["experiment_id"] == best_exp].sort_values("dimension")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(part["dimension"].astype(int), part["R2"], color="#7b68ee")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("qpos dimension")
            ax.set_ylabel("R2")
            ax.set_title(f"Per-joint qpos R2: best strict model ({best_exp})")
            ax.grid(alpha=0.25, axis="y")
            fig.tight_layout()
            fig.savefig(plot_dir / "strict_best_model_per_joint_qpos_r2.png", dpi=180)
            plt.close(fig)

            worst = part.sort_values("R2").head(10)
            bestj = part.sort_values("R2", ascending=False).head(10)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
            axes[0].barh(worst["dimension_name"], worst["R2"], color="#d95f02")
            axes[0].set_title("Worst qpos dimensions")
            axes[1].barh(bestj["dimension_name"], bestj["R2"], color="#1b9e77")
            axes[1].set_title("Best qpos dimensions")
            for ax in axes:
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("R2")
                ax.grid(alpha=0.25, axis="x")
            fig.suptitle("Best and worst joints for the top strict decoder")
            fig.tight_layout()
            fig.savefig(plot_dir / "strict_best_worst_joints_qpos_r2.png", dpi=180)
            plt.close(fig)

    # Overfitting check: train vs validation loss for sequence models.
    if not history.empty:
        top_history_ids = df.sort_values("test_qpos_R2", ascending=False).head(12)["experiment_id"].tolist()
        hist = history[history["experiment_id"].isin(top_history_ids)].copy()
        if not hist.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            for exp, group in hist.groupby("experiment_id"):
                label = exp.replace("repeat_holdout__", "").replace("__", " | ")
                axes[0].plot(group["epoch"], group["train_loss"], linewidth=1.1, alpha=0.75, label=label[:55])
                axes[1].plot(group["epoch"], group["val_loss"], linewidth=1.1, alpha=0.75, label=label[:55])
            axes[0].set_title("Training loss")
            axes[1].set_title("Validation loss")
            for ax in axes:
                ax.set_xlabel("epoch")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("loss")
            axes[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
            fig.suptitle("Overfitting check for top sequence models")
            fig.tight_layout()
            fig.savefig(plot_dir / "strict_top_sequence_train_val_loss.png", dpi=180)
            plt.close(fig)


def write_strict_report(out_dir: Path, leaderboard: pd.DataFrame, trial_table: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write a concise but thorough reviewer-facing README for strict results."""
    best_cols = [
        "experiment_id",
        "split_mode",
        "model_family",
        "policy_label",
        "target_base_mode",
        "pca_condition",
        "test_R2",
        "test_CC",
        "test_RMSE",
        "test_qpos_R2",
        "test_qpos_CC",
        "test_qpos_RMSE",
        "test_procrustes_corr",
        "test_qpos_procrustes_corr",
        "test_random_neural_R2",
        "test_random_neural_qpos_R2",
        "test_random_object_R2",
        "latency_ms_per_bin_offline",
        "train_ms",
        "mujoco_compare_dir",
        "mujoco_compare_1euro_dir",
    ]
    best_cols = [c for c in best_cols if c in leaderboard.columns]
    neural_pca_rows = leaderboard[leaderboard["pca_mode"].astype(str).str.startswith("pca")]
    target_pca_rows = leaderboard[leaderboard["target_pca"] == True]  # noqa: E712

    lines = [
        "# Strict Same-Threshold Direct Decoder Results",
        "",
        "This run was designed to answer the strict no-test-kinematics question. The evaluated models do not receive held-out qpos, qvel, ctrl, or training-set kinematic templates as input during validation/test. The only optional non-neural task variable is the object ID, which is assumed to be supplied online by the task controller as an integer.",
        "",
        "## What changed from the earlier hybrid run",
        "",
        "- Earlier `hybrid` rows used neural activity plus elapsed phase plus an object-specific qpos template built from training kinematics. That was useful as a task-assisted upper-bound style decoder, but it was not a purely neural direct decoder.",
        "- This strict suite defaults to direct decoders only. Inputs are `neural + phase` or `neural + phase + object_id_onehot`.",
        "- No held-out test kinematic trajectory is used as a model input. For qvel-only models, the first qpos sample is used only to reconstruct qpos for offline MuJoCo replay, standing in for the robot's known current state at trial onset.",
        "",
        "## Data used",
        "",
        "- The intended VM data root is `/mnt/vmshare`.",
        "- Folders directly under `/mnt/vmshare` are used.",
        "- The nested folder `/mnt/vmshare/thresholds defined per block` is intentionally ignored for this same-threshold rerun.",
        "- Only attempted-movement data are used. Imagine/observe/flag pickles are ignored for training.",
        "- Sampling is interpreted as 100 Hz, so one row/bin corresponds to 10 ms.",
        "",
        "Loaded sessions:",
        "",
        markdown_table(trial_table.groupby(["session"]).agg(trials=("repeat", "size"), gestures=("gesture", "nunique"), handles=("handle_id", "nunique")).reset_index()),
        "",
        "Loaded object counts:",
        "",
        markdown_table(trial_table.groupby(["object_name", "object_id", "object_label_source"]).size().reset_index(name="trials")),
        "",
        "## Object mapping",
        "",
        "The same-threshold VM pickles do not contain an explicit `Object.Id` per trial. The loader therefore uses the object mapping defined by the experimental protocol: handle IDs are mapped to MuJoCo geom/body IDs, then to six canonical object names.",
        "",
        markdown_table(pd.DataFrame({"handle_id": list(PROTOCOL_HANDLE_TO_GEOM_ID.keys()), "geom_id": list(PROTOCOL_HANDLE_TO_GEOM_ID.values()), "object_name": [VM_BODY_ID_TO_OBJECT[g] for g in PROTOCOL_HANDLE_TO_GEOM_ID.values()]})),
        "",
        "For each date/block, the known handle order selects the object label for each of the six gesture groups. The object ID passed to object-conditioned models is then a compact integer label: `0=small_ball`, `1=big_ball`, `2=small_v_cylinder`, `3=big_v_cylinder`, `4=small_h_cylinder`, `5=big_h_cylinder`.",
        "",
        "## Splits",
        "",
        "- `repeat_holdout`: train on repetitions 0-5, validate on repetitions 6-7, test on repetitions 8-9. This asks whether later repeats of known movements can be decoded.",
        "- `gesture_holdout`: hold out entire gesture/handle IDs. This asks whether a model generalizes to movements not seen during training.",
        "- `session_holdout`: hold out one full recording date. This asks whether a model transfers across days.",
        "- `block_holdout`: train on two blocks and test on one block. This asks whether a model transfers across sections of the same experimental protocol.",
        "",
        "## PCA conditions",
        "",
        "- `no N-PCA, no Y-PCA`: no PCA anywhere.",
        f"- `N-PCA, no Y-PCA`: PCA is fit on training neural bins only and the first {args.pca_components} neural components are fed to the model.",
        f"- `no N-PCA, Y-PCA`: the model predicts the first {args.target_pca_components} target PCs; the target is reconstructed before scoring and MuJoCo export.",
        f"- `N-PCA, Y-PCA`: both transformations are used, each fit only on the training split.",
        "",
        "Neural PCA is applied to the 256-channel neural feature vector after causal EMA smoothing. Target PCA is applied to the supervised output (`qpos`, `qvel`, or `qpos+qvel`), not to the model input. The PCA plots in `pca_diagnostics/` show how much training variance is captured by the chosen number of components.",
        "",
        "## Models",
        "",
        "- Ridge: a linear readout. All time bins are stacked into one matrix and a regularized linear map is fit from features to target.",
        "- Kalman: a classical state-space decoder. The hidden state is qpos+qvel; neural features are treated as observations. It can be scored as qpos, qvel, or qpos+qvel, but target PCA is not applied because the state remains physical qpos+qvel.",
        "- ReFIT Kalman: same Kalman framework, but training velocities are redirected toward the final target posture. This is an offline ReFIT-style approximation.",
        "- GRU: a causal recurrent neural network. It reads the movement history in order and predicts one target vector per 10 ms bin.",
        "- LFADS-lite: a compact supervised LFADS-like model. A recurrent encoder creates a low-dimensional factor state and a readout predicts the target; an auxiliary reconstruction loss encourages the factors to preserve neural dynamics.",
        "- Transformer: a causal two-layer self-attention model. Each bin can attend to past/current bins, not future bins.",
        "",
        "## Metrics",
        "",
        "- `test_R2`, `test_CC`, `test_RMSE`: metrics in the trained target space. For qvel rows, these describe velocity decoding; for qpos+qvel rows, they describe both position and velocity together.",
        "- `test_qpos_R2`, `test_qpos_CC`, `test_qpos_RMSE`: qpos-only metrics after converting the model output into qpos for MuJoCo replay. These are the most comparable across qpos, qvel, and qpos+qvel targets.",
        "- `test_procrustes_corr`: centered, scale-normalized target-trajectory shape similarity. Higher is better.",
        "- `test_qpos_procrustes_corr`: the same shape metric after conversion to qpos.",
        "- `onset_delay_s` / `offset_delay_s`: estimated start/end timing delay from movement-energy thresholding.",
        "- `jitter`: frame-to-frame roughness of the decoded trajectory; lower is smoother.",
        "- `latency_ms_per_bin_offline`: median inference wall time divided by decoded bins. At 100 Hz, the target budget is below 10 ms/bin before downstream communication overhead.",
        "- `test_random_neural_*`: the same decoder tested with random neural input. These scores should drop strongly if the model truly uses neural structure.",
        "- `test_random_object_*`: object-conditioned decoders tested with wrong/random object labels. This shows how much the object cue matters.",
        "",
        "## Best rows by qpos replay score",
        "",
        markdown_table(leaderboard.sort_values("test_qpos_R2", ascending=False)[best_cols], max_rows=35),
        "",
        "## PCA retained variance",
        "",
    ]
    if not neural_pca_rows.empty:
        neural_summary = neural_pca_rows.groupby(["split_mode", "pca_mode"], as_index=False)["pca_explained_variance"].first()
        neural_summary["explained_percent"] = 100.0 * neural_summary["pca_explained_variance"]
        lines.extend(["Neural PCA:", "", markdown_table(neural_summary), ""])
    if not target_pca_rows.empty:
        target_summary = target_pca_rows.groupby(["split_mode", "target_base_mode", "target_mode"], as_index=False)["target_pca_explained_variance"].first()
        target_summary["explained_percent"] = 100.0 * target_summary["target_pca_explained_variance"]
        lines.extend(["Target PCA:", "", markdown_table(target_summary), ""])
    lines.extend(
        [
            "## Main plots",
            "",
            "- `plots/strict_direct_target_r2_by_split.png`: compact strict-model comparison. Target R2 is on the y-axis; split type is on the x-axis; color encodes model; marker encodes whether object ID was used; lighter/smaller points indicate target PCA.",
            "- `plots/strict_best_qpos_replay_r2.png`: best qpos-replay rows, useful because it compares qpos, qvel, and qpos+qvel models in the common final output space.",
            "- `plots/strict_top_qpos_r2_mean_std.png`: mean and standard deviation of qpos R2 across trials for top models.",
            "- `plots/strict_best_model_object_qpos_r2.png`: object-wise qpos R2 for the best model per split.",
            "- `plots/strict_session_variance_top_rows.png`: session/date variability for top rows.",
            "- `plots/strict_best_model_per_joint_qpos_r2.png`: R2 per qpos dimension for the best strict model.",
            "- `plots/strict_best_worst_joints_qpos_r2.png`: easiest and hardest qpos dimensions for the best strict model.",
            "- `plots/strict_top_sequence_train_val_loss.png`: train/validation loss curves for top GRU/LFADS/Transformer rows, used to inspect overfitting.",
            "- `pca_diagnostics/*neural_pca_explained_variance.png`: neural PCA scree curves.",
            "- `pca_diagnostics/*target_pca_explained_variance.png`: target PCA scree curves.",
            "",
            "## MuJoCo replay",
            "",
            "Every row writes two replay folders: raw decoded qpos and 1Euro-filtered decoded qpos. The exported trials are selected using the same repeat/session/handle filters across models, so different decoders can be compared on the same examples whenever the split contains that trial.",
            "",
            "Example after downloading a result folder to the Mac:",
            "",
            "```bash",
            "cd \"<project folder>\"",
            "mjpython scripts/view_mujoco_hand.py \\",
            "  --compare \\",
            "  --compare-dir \"/absolute/path/to/result/mujoco_compare/<experiment_id>\" \\",
            "  --repeat 8 \\",
            "  --camera-view arm \\",
            "  --factor 120 \\",
            "  --slow-loop-factor 1200 \\",
            "  --loop",
            "```",
            "",
            "To make a minimal replay zip for someone else:",
            "",
            "```bash",
            "python scripts/package_mujoco_replay_minimal.py \\",
            "  --result-dir /absolute/path/to/result_folder \\",
            "  --experiment-id <experiment_id> \\",
            "  --out-zip /absolute/path/to/mujoco_replay_minimal.zip",
            "```",
            "",
            "## BRAND note",
            "",
            "The BRAND node should be configured through YAML with the chosen saved artifact path. For strict online deployment, use a causal GRU or LFADS-lite direct row with `repeat_holdout`, `target=qpos` or `target=qpos+qvel`, no target PCA unless the artifact loader has been validated on site, and no bidirectional LFADS. The node receives one 256-channel neural bin plus a decode/rest flag every 10 ms. If the flag is off, no qpos packet is emitted. If the flag is on, the node emits one 32-D qpos command and latency metadata.",
            "",
        ]
    )
    (out_dir / "STRICT_TRAINING_RESULTS_README.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_experiment(
    *,
    experiment_id: str,
    model_family: str,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector,
    n_objects: int,
    condition_object: bool,
    target_codec: TargetCodec,
    split_mode: str,
    pca_mode: str,
    device: str,
    profile: str,
    seed: int,
    bidirectional_lfads: bool,
    gru_layers: int,
    sample_rate_hz: float,
    out_dir: Path,
    export_repeat: int | None,
    export_session: str,
    export_handle_id: int | None,
    export_max_trials: int,
    qpos_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Train/evaluate one strict configuration and save model/replay outputs."""
    model_out = out_dir / "models" / experiment_id
    model_out.mkdir(parents=True, exist_ok=True)
    if model_family == "ridge":
        row, preds_target, _predict_fn, _train_ms = train_eval_ridge_direct(
            train,
            val,
            test,
            projector,
            n_objects,
            condition_object=condition_object,
            target_codec=target_codec,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
        )
        history = pd.DataFrame()
    elif model_family in {"gru", "lfads_lite", "transformer"}:
        row, preds_target, _predict_fn, history = train_sequence_direct(
            train,
            val,
            test,
            projector,
            n_objects,
            model_family=model_family,
            condition_object=condition_object,
            target_codec=target_codec,
            device_name=device,
            profile=profile,
            seed=seed,
            bidirectional_lfads=bidirectional_lfads,
            gru_layers=gru_layers,
            sample_rate_hz=sample_rate_hz,
            out_dir=model_out,
        )
    elif model_family in {"kalman", "refit_kalman"}:
        row, preds_target, _predict_fn = train_eval_kalman_direct(
            train,
            val,
            test,
            projector,
            model_family=model_family,
            base_mode=target_codec.base_mode,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
            n_objects=n_objects,
        )
        history = pd.DataFrame()
    else:
        raise ValueError(model_family)

    row = {
        **row,
        "experiment_id": experiment_id,
        "split_mode": split_mode,
        "pca_mode": pca_mode,
        "pca_explained_variance": projector.explained_variance,
        "target_mode": target_codec.mode,
        "target_base_mode": target_codec.base_mode,
        "target_pca": bool(target_codec.uses_pca),
        "target_pca_components": target_codec.output_dim if target_codec.uses_pca else np.nan,
        "target_pca_explained_variance": target_codec.explained_variance if target_codec.uses_pca else np.nan,
        "n_train_trials": len(train),
        "n_val_trials": len(val),
        "n_test_trials": len(test),
    }
    row["policy_label"] = policy_label(row)
    row["pca_condition"] = pca_condition_label(row)

    qpos_preds = qpos_predictions_from_targets(test, preds_target, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz)
    filtered_qpos_preds = one_euro_filter_trials(qpos_preds, sample_rate_hz=sample_rate_hz)
    filtered_metrics = qpos_metrics_only(test, filtered_qpos_preds, sample_rate_hz=sample_rate_hz)
    row.update({f"test_1euro_qpos_{k}": v for k, v in filtered_metrics.items()})

    append_trial_metrics(
        out_dir,
        experiment_id=experiment_id,
        row=row,
        trials=test,
        preds_target=preds_target,
        qpos_preds=qpos_preds,
        sample_rate_hz=sample_rate_hz,
        output_variant="raw",
    )
    append_trial_metrics(
        out_dir,
        experiment_id=experiment_id,
        row=row,
        trials=test,
        preds_target=preds_target,
        qpos_preds=filtered_qpos_preds,
        sample_rate_hz=sample_rate_hz,
        output_variant="1euro",
    )
    append_per_dimension_metrics(
        out_dir,
        experiment_id=experiment_id,
        row=row,
        trials=test,
        preds_target=preds_target,
        qpos_preds=qpos_preds,
        qpos_names=qpos_names,
    )

    compare_dir = out_dir / "mujoco_compare" / experiment_id
    export_mujoco_compare_selected(
        test,
        qpos_preds,
        compare_dir,
        experiment_id,
        max_trials=export_max_trials,
        export_repeat=export_repeat,
        export_session=export_session,
        export_handle_id=export_handle_id,
    )
    row["mujoco_compare_dir"] = str(compare_dir)
    filtered_compare_dir = out_dir / "mujoco_compare_1euro" / experiment_id
    export_mujoco_compare_selected(
        test,
        filtered_qpos_preds,
        filtered_compare_dir,
        f"{experiment_id}__1euro",
        max_trials=export_max_trials,
        export_repeat=export_repeat,
        export_session=export_session,
        export_handle_id=export_handle_id,
    )
    row["mujoco_compare_1euro_dir"] = str(filtered_compare_dir)

    if not history.empty:
        history["experiment_id"] = experiment_id
        history["split_mode"] = split_mode
        history["pca_mode"] = pca_mode
        history["pca_condition"] = row["pca_condition"]
    return clean_dict(row), history


def make_model_specs(model_families: list[str], condition_modes: list[str], profile: str) -> list[dict]:
    """Build the strict direct model grid."""
    specs = []
    for family in model_families:
        if family in {"kalman", "refit_kalman"}:
            specs.append({"model_family": family, "condition_object": False})
            continue
        for condition in condition_modes:
            specs.append({"model_family": family, "condition_object": condition == "obj"})
    if profile == "smoke":
        # Smoke should validate data, splits, metrics, plotting, and export
        # quickly.  The full command below runs the entire grid.
        keep = []
        for spec in specs:
            if spec["model_family"] in {"ridge", "gru"} and not spec["condition_object"]:
                keep.append(spec)
        return keep[:2] if keep else specs[:2]
    return specs


def write_analysis_notebook(out_dir: Path) -> None:
    """Create an interpretation notebook for the strict result folder."""
    notebook_dir = out_dir / "analysis"
    notebook_dir.mkdir(parents=True, exist_ok=True)

    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

    cells = [
        md(
            "# Strict same-threshold decoder analysis\n\n"
            "This notebook reloads the CSV outputs from the strict VM suite. Training has already happened; this is for interpretation, thesis figures, sanity checks, and choosing which MuJoCo replay to show.\n\n"
            "The central rule for this run is that no held-out qpos, qvel, ctrl, or template trajectory was fed into the decoder during validation/test."
        ),
        code(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "RESULT_DIR = Path('..').resolve()\n"
            "leaderboard = pd.read_csv(RESULT_DIR / 'leaderboard.csv')\n"
            "trial_metrics = pd.read_csv(RESULT_DIR / 'trial_metrics.csv')\n"
            "per_dim = pd.read_csv(RESULT_DIR / 'per_dimension_metrics.csv')\n"
            "history_path = RESULT_DIR / 'training_history.csv'\n"
            "history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()\n"
            "print(RESULT_DIR)\n"
            "print('leaderboard rows:', len(leaderboard))\n"
            "print('trial metric rows:', len(trial_metrics))\n"
            "print('per-dimension rows:', len(per_dim))\n"
        ),
        md(
            "## 1. Best rows\n\n"
            "`test_R2` is the score in the trained target space. If the model was trained on qvel, `test_R2` is qvel R2. `test_qpos_R2` is always the reconstructed qpos score, so it is the easiest number to compare across qpos, qvel, and qpos+qvel rows."
        ),
        code(
            "cols = ['split_mode','model_family','policy_label','target_base_mode','pca_condition','test_R2','test_qpos_R2','test_qpos_CC','test_qpos_RMSE','latency_ms_per_bin_offline']\n"
            "leaderboard.sort_values('test_qpos_R2', ascending=False)[cols].head(30)\n"
        ),
        md(
            "## 2. Object label effect\n\n"
            "This table compares object-conditioned rows against no-label rows. If object rows improve strongly, the decoder benefits from knowing which object is instructed. If no-label rows are close, the neural activity and phase already carry most of the useful information."
        ),
        code(
            "obj_effect = (leaderboard\n"
            "    .groupby(['split_mode','target_base_mode','model_family','policy_label'], as_index=False)['test_qpos_R2']\n"
            "    .max()\n"
            "    .pivot_table(index=['split_mode','target_base_mode','model_family'], columns='policy_label', values='test_qpos_R2'))\n"
            "obj_effect['object_minus_nolabel'] = obj_effect.get('object') - obj_effect.get('no label')\n"
            "obj_effect.sort_values('object_minus_nolabel', ascending=False).head(30)\n"
        ),
        md(
            "## 3. PCA effect\n\n"
            "Neural PCA compresses the 256-channel input after causal smoothing. Target PCA compresses qpos/qvel/qpos+qvel outputs. Both are fit only on training rows. This table asks whether compression helps or hurts qpos replay quality."
        ),
        code(
            "pca_effect = (leaderboard\n"
            "    .groupby(['split_mode','target_base_mode','model_family','pca_condition'], as_index=False)['test_qpos_R2']\n"
            "    .max()\n"
            "    .sort_values(['split_mode','target_base_mode','test_qpos_R2'], ascending=[True, True, False]))\n"
            "pca_effect.head(40)\n"
        ),
        md(
            "## 4. Random controls\n\n"
            "`test_random_neural_*` uses the trained decoder but replaces the test neural bins with random samples drawn from the training neural distribution. Good decoders should get worse. `test_random_object_*` randomizes the object label for object-conditioned models; a large drop means the object cue matters."
        ),
        code(
            "control_cols = ['split_mode','model_family','policy_label','target_base_mode','pca_condition','test_qpos_R2','test_random_neural_qpos_R2','test_random_object_qpos_R2']\n"
            "leaderboard[control_cols].sort_values('test_qpos_R2', ascending=False).head(30)\n"
        ),
        md(
            "## 5. Trial variability, objects, and sessions\n\n"
            "Trial variability is summarized with mean and standard deviation. The per-trial table lets us summarize variability by object, session/date, block, gesture, and model condition."
        ),
        code(
            "raw = trial_metrics[trial_metrics['output_variant'].eq('raw')].copy()\n"
            "best_exp = leaderboard.sort_values('test_qpos_R2', ascending=False).iloc[0]['experiment_id']\n"
            "best_trials = raw[raw['experiment_id'].eq(best_exp)]\n"
            "display(best_exp)\n"
            "display(best_trials.groupby('object_name')['qpos_R2'].agg(['mean','std','count']).sort_values('mean'))\n"
            "display(best_trials.groupby('session_date')['qpos_R2'].agg(['mean','std','count']))\n"
        ),
        md(
            "## 6. Per-joint performance\n\n"
            "This identifies which qpos dimensions are easiest and hardest for the best qpos-replay model. It is useful for deciding whether shoulder/elbow/wrist or fingers are decoded better."
        ),
        code(
            "qpos_dim = per_dim[(per_dim['space'].eq('qpos_replay')) & (per_dim['experiment_id'].eq(best_exp))].copy()\n"
            "display(qpos_dim.sort_values('R2').head(12)[['dimension','dimension_name','R2','CC','RMSE']])\n"
            "display(qpos_dim.sort_values('R2', ascending=False).head(12)[['dimension','dimension_name','R2','CC','RMSE']])\n"
        ),
        md(
            "## 7. Overfitting\n\n"
            "A healthy neural model usually has train and validation loss decreasing together. A widening gap means the network is memorizing training trials. Early stopping selects the best validation epoch, not the best test epoch."
        ),
        code(
            "if not history.empty:\n"
            "    top_ids = leaderboard.sort_values('test_qpos_R2', ascending=False).head(10)['experiment_id'].tolist()\n"
            "    display(history[history['experiment_id'].isin(top_ids)].groupby('experiment_id')[['train_loss','val_loss']].tail(1))\n"
            "else:\n"
            "    print('No neural-network history file was saved for this run.')\n"
        ),
        md("## 8. Main generated plots"),
        code(
            "plot_names = [\n"
            "    'strict_direct_target_r2_by_split.png',\n"
            "    'strict_best_qpos_replay_r2.png',\n"
            "    'strict_top_qpos_r2_mean_std.png',\n"
            "    'strict_best_model_object_qpos_r2.png',\n"
            "    'strict_session_variance_top_rows.png',\n"
            "    'strict_best_model_per_joint_qpos_r2.png',\n"
            "    'strict_best_worst_joints_qpos_r2.png',\n"
            "    'strict_top_sequence_train_val_loss.png',\n"
            "]\n"
            "for name in plot_names:\n"
            "    path = RESULT_DIR / 'plots' / name\n"
            "    if path.exists():\n"
            "        print(name)\n"
            "        display(Image(filename=str(path)))\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (notebook_dir / "strict_same_thresholds_results_explainer.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Run the strict direct-output suite."""
    parser = argparse.ArgumentParser(description="Run strict neural-only qpos/qvel/pv decoder suite.")
    parser.add_argument("--out-dir", type=Path, default=BUNDLE_ROOT / "artifacts" / "strict_direct_output_suite")
    parser.add_argument("--data-root", type=Path, action="append", default=None)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--profile", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"],
        choices=["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"],
    )
    parser.add_argument("--pca-modes", nargs="+", default=["none", "pca"], choices=["none", "pca"])
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument(
        "--target-modes",
        nargs="+",
        default=["qpos", "qvel", "pv", "qpos_pca", "qvel_pca", "pv_pca"],
        help="Use qpos/qvel/pv plus *_pca variants to cover no PCA, neural PCA only, target PCA only, and both.",
    )
    parser.add_argument("--target-pca-components", type=int, default=DEFAULT_TARGET_PCA_COMPONENTS)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--bidirectional-lfads", action="store_true")
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument(
        "--model-families",
        nargs="+",
        default=["ridge", "kalman", "refit_kalman", "gru", "lfads_lite", "transformer"],
        choices=["ridge", "kalman", "refit_kalman", "gru", "lfads_lite", "transformer"],
    )
    parser.add_argument("--condition-modes", nargs="+", default=["nolabel", "obj"], choices=["nolabel", "obj"])
    parser.add_argument("--session-holdout-test-date", default="")
    parser.add_argument("--block-holdout-test-block", type=int, default=3)
    parser.add_argument("--export-repeat", type=int, default=8)
    parser.add_argument("--export-session", default="")
    parser.add_argument("--export-handle-id", type=int, default=None)
    parser.add_argument("--export-max-trials", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_roots = args.data_root if args.data_root is not None else [p for p in OLD_PATIENT_ROOTS if p.exists()]
    if not data_roots:
        raise SystemExit(
            "No --data-root folders were provided and no bundled patient_sessions were found. "
            "On the VM, run this through run_strict_same_thresholds_suite.sh so session folders "
            "under /mnt/vmshare are discovered and passed in automatically."
        )
    dataset, trial_table = combine_sessions(list(data_roots), args.template_dir, args.out_dir)
    trial_table.to_csv(args.out_dir / "combined_trial_index.csv", index=False)
    n_objects = int(max(t.object_id for t in dataset.trials) + 1)
    qpos_names = list(getattr(dataset, "qpos_names", []) or getattr(dataset, "joint_names", []) or [f"qpos_{i}" for i in range(N_QPOS)])

    rows = []
    histories = []
    experiment_count = 0
    target_modes = [canonical_target_mode(m) for m in args.target_modes]
    model_specs = make_model_specs(args.model_families, args.condition_modes, args.profile)

    for split_mode in args.splits:
        train, val, test, split_info = make_split(
            dataset.trials,
            split_mode,
            args.seed,
            session_holdout_test_date=args.session_holdout_test_date,
            block_holdout_test_block=args.block_holdout_test_block,
        )
        (args.out_dir / f"split_{split_mode}.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
        if "pca" in args.pca_modes:
            save_neural_pca_diagnostics(
                train,
                split_mode=split_mode,
                out_dir=args.out_dir,
                marker_components=args.pca_components,
                seed=args.seed,
            )
        for base in sorted({target_base_mode(m) for m in target_modes}):
            save_target_pca_diagnostics(
                train,
                split_mode=split_mode,
                base_mode=base,
                out_dir=args.out_dir,
                marker_components=args.target_pca_components,
                seed=args.seed,
            )

        for pca_mode in args.pca_modes:
            projector = fit_neural_projector(
                train,
                mode="none" if pca_mode == "none" else "pca",
                components=args.pca_components,
                seed=args.seed,
            )
            for target_mode in target_modes:
                target_codec = fit_target_codec(
                    train,
                    requested_mode=target_mode,
                    components=args.target_pca_components,
                    seed=args.seed + experiment_count,
                )
                for spec in model_specs:
                    family = spec["model_family"]
                    condition_object = bool(spec["condition_object"])
                    if family in {"kalman", "refit_kalman"} and target_codec.uses_pca:
                        print("Skipping target-PCA Kalman/ReFIT row because the Kalman state stays physical qpos+qvel.", flush=True)
                        continue
                    experiment_id = (
                        f"{split_mode}__{projector.mode}__{family}__direct__"
                        f"{'obj' if condition_object else 'nolabel'}__target_{target_codec.mode}"
                    )
                    experiment_count += 1
                    if args.max_experiments is not None and experiment_count > args.max_experiments:
                        break
                    print(f"\n=== {experiment_id} ===", flush=True)
                    row, history = run_one_experiment(
                        experiment_id=experiment_id,
                        model_family=family,
                        train=train,
                        val=val,
                        test=test,
                        projector=projector,
                        n_objects=n_objects,
                        condition_object=condition_object,
                        target_codec=target_codec,
                        split_mode=split_mode,
                        pca_mode=projector.mode,
                        device=device,
                        profile=args.profile,
                        seed=args.seed + experiment_count,
                        bidirectional_lfads=args.bidirectional_lfads,
                        gru_layers=args.gru_layers,
                        sample_rate_hz=args.sample_rate_hz,
                        out_dir=args.out_dir,
                        export_repeat=args.export_repeat,
                        export_session=args.export_session,
                        export_handle_id=args.export_handle_id,
                        export_max_trials=args.export_max_trials,
                        qpos_names=qpos_names,
                    )
                    rows.append(row)
                    if not history.empty:
                        histories.append(history)
                    pd.DataFrame(rows).to_csv(args.out_dir / "leaderboard_partial.csv", index=False)
                if args.max_experiments is not None and experiment_count >= args.max_experiments:
                    break
            if args.max_experiments is not None and experiment_count >= args.max_experiments:
                break
        if args.max_experiments is not None and experiment_count >= args.max_experiments:
            break

    leaderboard = pd.DataFrame(rows).sort_values(["test_qpos_R2", "test_R2"], ascending=False)
    leaderboard.to_csv(args.out_dir / "leaderboard.csv", index=False)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    if not history_df.empty:
        history_df.to_csv(args.out_dir / "training_history.csv", index=False)

    manifest = {
        "data_roots": [str(p) for p in data_roots],
        "splits": args.splits,
        "pca_modes": args.pca_modes,
        "pca_components": args.pca_components,
        "target_modes": target_modes,
        "target_pca_components": args.target_pca_components,
        "model_families": args.model_families,
        "condition_modes": args.condition_modes,
        "profile": args.profile,
        "device": device,
        "sample_rate_hz": args.sample_rate_hz,
        "bidirectional_lfads": bool(args.bidirectional_lfads),
        "gru_layers": int(args.gru_layers),
        "uses_falcon": False,
        "strict_no_template_hybrid": True,
        "uses_test_kinematic_input": False,
        "notes": (
            "Direct models only. Optional object labels come from session metadata/protocol mapping. "
            "No held-out qpos/qvel/ctrl or template trajectory is fed as decoder input."
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_clean_suite(args.out_dir, leaderboard, history_df, trial_table)
    write_strict_report(args.out_dir, leaderboard, trial_table, args)
    write_analysis_notebook(args.out_dir)
    print(f"Wrote strict suite outputs to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
