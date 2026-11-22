#!/usr/bin/env python3
"""Strict neural-only decoder suite for the same-threshold patient VM data.

This script is separate from ``run_split_pca_model_suite.py``.
That suite included *hybrid/template-assisted* decoders.  Those
hybrid rows did use training-set kinematic templates as input priors.  For the stricter
rerun we ended up evaluating for the thesis reported results, the default model grid excludes those template inputs.

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
    temporal_metrics_by_trial,
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
        return (qpos0 + np.vstack([np.zeros((1,N_QPOS),dtype=np.float32), np.cumsum(pred[:-1, :N_QPOS] * dt, axis=0)])).astype(np.float32)
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
    target_metrics = temporal_metrics_by_trial([target_array(t, base_mode)[:len(p)] for t,p in zip(trials,preds_target,strict=True)], preds_target, sample_rate_hz)

    qpos_preds = qpos_predictions_from_targets(trials, preds_target, base_mode=base_mode, sample_rate_hz=sample_rate_hz)
    true_qpos = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, qpos_preds, strict=True)], axis=0)
    pred_qpos = np.concatenate(qpos_preds, axis=0)
    qpos_metrics = temporal_metrics_by_trial([t.qpos[:len(p)] for t,p in zip(trials,qpos_preds,strict=True)], qpos_preds, sample_rate_hz)

    out = {**target_metrics}
    out.update({f"qpos_{k}": v for k, v in qpos_metrics.items()})
    return out


def qpos_metrics_only(trials: list[Trial], qpos_preds: list[np.ndarray], *, sample_rate_hz: float) -> dict[str, float]:
    """Compute metrics when predictions are already qpos trajectories."""
    true_qpos = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, qpos_preds, strict=True)], axis=0)
    pred_qpos = np.concatenate(qpos_preds, axis=0)
    return temporal_metrics_by_trial([t.qpos[:len(p)] for t,p in zip(trials,qpos_preds,strict=True)], qpos_preds, sample_rate_hz)


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
        row["selection_score"] = row["val_qpos_R2"]
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
        row["selection_score"] = row["val_qpos_R2"]
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
            "phase_mode": "fixed_clock_6s",
            "output_alpha": best_alpha,
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
    target_codec: TargetCodec,
    model_out: Path,
) -> tuple[dict, list[np.ndarray], object]:
    """Train/evaluate Kalman or ReFIT Kalman for qpos/qvel/pv outputs."""
    refit = model_family == "refit_kalman"
    states = [corrected_pv_state(t, refit=refit) for t in train]
    state_mean = None; state_components = None
    if target_codec.uses_pca:
        state_pca = PCA(n_components=min(12, states[0].shape[1]), svd_solver="randomized", random_state=seed)
        state_pca.fit(np.concatenate(states))
        state_mean = state_pca.mean_.astype(np.float32)
        state_components = state_pca.components_.astype(np.float32)
        states = [((z-state_mean) @ state_components.T).astype(np.float32) for z in states]
    obs = [neural_features(t, projector) for t in train]
    t0 = time.perf_counter()
    params = fit_kalman_matrices(states, obs, ridge=1e-2)
    train_ms = (time.perf_counter() - t0) * 1000.0

    def z_to_target(z: np.ndarray) -> np.ndarray:
        if state_components is not None:
            z = z @ state_components + state_mean
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
    row["selection_score"] = row["val_qpos_R2"]
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
    import pickle
    artifact = model_out / "kalman.pkl"
    with artifact.open("wb") as f:
        pickle.dump({'model_family': model_family, 'params': {k:v for k,v in params.items() if not k.startswith('_')},
                     'projector': {'mode': projector.mode, 'mean': projector.mean, 'components': projector.components}, 'state_mean': state_mean, 'state_components': state_components,
                     'target_base_mode': base_mode, 'sample_rate_hz': sample_rate_hz,
                     'phase_mode': 'not_used'}, f)
    row['artifact'] = str(artifact)
    row['state_pca_components'] = 0 if state_components is None else len(state_components)
    row['state_pca_explained_variance'] = None if state_components is None else float(state_pca.explained_variance_ratio_.sum())
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
        actual_qpos_ctrl = qpos_to_ctrl(trial.qpos[:n], metadata.ctrl_to_qpos_index)
        np.save(actual_path, actual_qpos_ctrl.T.astype(np.float32))
        recorded_ctrl_path = out_dir / f"{stem}_recorded_ctrl.npy"
        np.save(recorded_ctrl_path, trial.ctrl[:n].T.astype(np.float32))
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
                "recorded_ctrl": str(recorded_ctrl_path),
                "decoded": str(decoded_path),
            }
        )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": model_label,
                "actual_output": "reference qpos mapped to ctrl; original recorded ctrl is saved separately",
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
        model, _ = train_ridge_direct(train, projector, n_objects, condition_object=condition_object, target_codec=target_codec, alpha=100.0)
        import pickle
        artifact = model_out / "ridge.pkl"
        with artifact.open("wb") as f:
            pickle.dump({'model_family': 'ridge', 'x_mean': model.x_scaler.mean, 'x_scale': model.x_scaler.scale, 'y_mean': model.y_scaler.mean, 'y_scale': model.y_scaler.scale, 'weights': model.weights, 'bias': model.bias, 'condition_object': condition_object, 'target_mean': target_codec.mean, 'target_components': target_codec.components, 'target_base_mode': target_codec.base_mode, 'projector': {'mode': projector.mode, 'mean': projector.mean, 'components': projector.components},
                         'n_objects': n_objects, 'output_alpha': row['output_alpha'],
                         'phase_mode': 'fixed_clock_6s', 'sample_rate_hz': sample_rate_hz}, f)
        row['artifact'] = str(artifact)
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
            target_codec=target_codec,
            model_out=model_out,
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
    if model_family in {"kalman","refit_kalman"}:
        row['target_pca'] = False
        row['target_pca_components'] = None
        row['target_pca_explained_variance'] = None
        row['target_mode'] = target_codec.base_mode + ('_statepca12' if target_codec.uses_pca else '')
    row["policy_label"] = policy_label(row)
    row["pca_condition"] = pca_condition_label(row)

    qpos_preds = qpos_predictions_from_targets(test, preds_target, base_mode=target_codec.base_mode, sample_rate_hz=sample_rate_hz)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    lengths = np.asarray([len(p) for p in qpos_preds], dtype=np.int32)
    np.savez_compressed(prediction_dir / (experiment_id + '.npz'), lengths=lengths,
                        qpos_true=np.concatenate([t.qpos[:len(p)] for t,p in zip(test,qpos_preds,strict=True)]),
                        qpos_pred=np.concatenate(qpos_preds), target_pred=np.concatenate(preds_target),
                        target_true=np.concatenate([target_array(t,target_codec.base_mode)[:len(p)] for t,p in zip(test,preds_target,strict=True)]))
    row['prediction_archive'] = str(prediction_dir / (experiment_id+'.npz'))
    row['experiment_seed'] = int(seed)
    row['amortized_offline_ms_per_bin'] = row['latency_ms_per_bin_offline']
    row['temporal_aggregation'] = 'within-trial metrics; equal-trial mean'
    row['phase_mode'] = 'not_used' if model_family in {'kalman','refit_kalman'} else 'fixed_clock_6s'
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
    return specs




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
            "On the VM, run this through run_final.py so session folders "
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
                    seed=args.seed,
                )
                for spec in model_specs:
                    family = spec["model_family"]
                    condition_object = bool(spec["condition_object"])
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
                        seed=args.seed + int.from_bytes(__import__("hashlib").sha256(experiment_id.encode()).digest()[:4], "little") % 100000,
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

    leaderboard = pd.DataFrame(rows).sort_values("val_qpos_R2", ascending=False)
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
        "phase_mode": "fixed_clock_6s",
        "phase_description": "bin_index / 600; no trial-length input; no clipping",
        "selection_rule": "maximum validation qpos R2",
        "base_seed": args.seed,
        "experiment_seed_rule": "base_seed + little-endian first four SHA256 bytes of experiment_id modulo 100000",
        "velocity_integration": "q[0]=known initial state; q[t]=q[0]+dt*sum(v[:t])",
        "kalman_pca_description": "training PV-state PCA12, not an output-head PCA",
        "strict_no_template_hybrid": True,
        "uses_test_kinematic_input": False,
        "notes": (
            "Direct models only. Optional object labels come from session metadata/protocol mapping. "
            "No held-out qpos/qvel/ctrl or template trajectory is fed as decoder input."
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_group_metric_summaries(args.out_dir)
    print(f"Wrote strict suite outputs to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
