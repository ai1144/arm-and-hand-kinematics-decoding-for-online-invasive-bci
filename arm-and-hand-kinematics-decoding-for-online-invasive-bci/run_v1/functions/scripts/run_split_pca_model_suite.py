#!/usr/bin/env python3
"""Train/evaluate the main patient decoder grid.

This is the central VM training script, BUT IT CONTAINS THE HYBRID DECODERS.
For direct decoders (i.e.: inout is only neural data + object ID + elapsed time), please check /run_strict_direct_output_suite.py  
This script receives already-selected ``--data-root`` folders from ``run_vm_split_pca_suite.py`` and trains the
decoder families used in the thesis comparison:

- Ridge, Kalman, ReFIT-style Kalman
- causal GRU
- LFADS-lite
- causal Transformer

The input data are patient attempt trials.  After loading, every trial has the
same simple shape, regardless of whether it came from the old local sessions on my machine or
the newer VM blocks:

- ``trial.neural``: ``(time, 256)`` binned neural activity.
- ``trial.qpos``: ``(time, 32)`` target robot joint positions, in qpos order.
- ``trial.qvel``: ``(time, 32)`` joint velocities, mainly used by Kalman state.
- ``trial.ctrl``: ``(time, 32)`` MuJoCo control-order values for replay only.
- labels such as ``gesture``, ``repeat``, ``handle_id``, ``object_id``.

The main supervised target is always 32-D qpos at 100 Hz.  A direct decoder
learns ``neural features -> qpos``.  A hybrid decoder is task-assisted: it gets
an object/template prior and learns only the residual ``qpos - template``.  PCA,
when enabled, is fit on training neural bins only and is then frozen for
validation/test so no held-out data leak into preprocessing.

MuJoCo export files are converted from qpos order to ctrl order only for
visualization; that conversion does not change what the decoder was trained to
predict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder_core.data import Trial, causal_ema, split_trials, PROTOCOL_HANDLE_TO_GEOM_ID, VM_BODY_ID_TO_OBJECT
from decoder_core.metrics import mean_corr, per_dim_r2, rmse, summarize
from decoder_core.models import Standardizer, _fit_ridge_readout
from decoder_core.mujoco_metadata import DEFAULT_MUJOCO_XML, parse_mujoco_metadata
from run_multisession_online_decoding import combine_sessions
from train_smooth_decoders import smooth_output


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OLD_PATIENT_ROOTS = [
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0116",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0123",
    BUNDLE_ROOT / "patient_sessions" / "MM_sample_data_0130",
]
DEFAULT_TEMPLATE_DIR = BUNDLE_ROOT / "new_info" / "data"
N_QPOS = 32
DT = 0.01  # default VM timing: 100 Hz, so one bin is 10 ms.
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0


def clean_value(value):
    """Convert numpy/path values into JSON/CSV-friendly Python values."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    return value


def clean_dict(row: dict) -> dict:
    """Clean every value in one result row before saving it."""
    return {k: clean_value(v) for k, v in row.items()}


class OneEuroFilter:
    """Small vectorized 1Euro filter used only for optional output smoothing.

    The training target and raw leaderboard metrics are unchanged.  This filter
    is applied after decoding to create an additional MuJoCo replay folder and
    additional `test_1euro_*` metric columns.
    """

    def __init__(self, *, min_cutoff: float, beta: float, d_cutoff: float, dt: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dt = float(dt)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | float = 0.0

    def alpha(self, cutoff: np.ndarray | float) -> np.ndarray | float:
        """Convert cutoff frequency into exponential smoothing alpha."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self.dt)

    def update(self, x: np.ndarray) -> np.ndarray:
        """Filter one 32-D qpos row using current and previous rows only."""
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            return x.copy()
        dx = (x - self.x_prev) / self.dt
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat.astype(np.float32)
        self.dx_prev = dx_hat.astype(np.float32)
        return self.x_prev.copy()


def one_euro_filter_trajectory(
    y: np.ndarray,
    *,
    sample_rate_hz: float,
    min_cutoff: float = ONE_EURO_MIN_CUTOFF,
    beta: float = ONE_EURO_BETA,
    d_cutoff: float = ONE_EURO_D_CUTOFF,
) -> np.ndarray:
    """Apply a causal 1Euro filter to one decoded qpos trajectory."""
    filt = OneEuroFilter(
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff,
        dt=1.0 / float(sample_rate_hz),
    )
    out = np.empty_like(np.asarray(y, dtype=np.float32))
    for i in range(out.shape[0]):
        out[i] = filt.update(y[i])
    return out.astype(np.float32)


def one_euro_filter_trials(preds: list[np.ndarray], *, sample_rate_hz: float) -> list[np.ndarray]:
    """Apply the 1Euro filter independently to every decoded test trial."""
    return [one_euro_filter_trajectory(pred, sample_rate_hz=sample_rate_hz) for pred in preds]


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a dataframe as a simple Markdown table for result READMEs."""
    if df.empty:
        return "_No rows._"
    show = df if max_rows is None else df.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def resample_rows(values: np.ndarray, n_rows: int) -> np.ndarray:
    """Resample a trajectory to a fixed number of rows using phase interpolation."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == n_rows:
        return values.copy()
    old_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, n_rows, dtype=np.float32)
    out = np.empty((n_rows, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(new_x, old_x, values[:, col]).astype(np.float32)
    return out


def deterministic_gesture_split(
    trials: list[Trial],
    *,
    seed: int,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> tuple[list[Trial], list[Trial], list[Trial], dict]:
    """Create a reproducible unseen-gesture split by holding out gesture IDs."""
    # Gesture holdout is the strict split: every repeat of selected gestures is
    # excluded from training.  This tests whether a decoder can generalize to
    # movement conditions that were never represented in the training set.
    gestures = np.asarray(sorted({int(t.gesture) for t in trials}), dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = gestures.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    test_g = set(int(x) for x in shuffled[:n_test])
    val_g = set(int(x) for x in shuffled[n_test : n_test + n_val])
    train_g = set(int(x) for x in shuffled[n_test + n_val :])
    train = [t for t in trials if int(t.gesture) in train_g]
    val = [t for t in trials if int(t.gesture) in val_g]
    test = [t for t in trials if int(t.gesture) in test_g]
    return train, val, test, {
        "train_gestures": sorted(train_g),
        "val_gestures": sorted(val_g),
        "test_gestures": sorted(test_g),
    }


def session_date_from_name(name: str) -> str:
    """Extract YYYYMMDD from a session/block folder name."""
    import re

    match = re.search(r"(20\d{6})", str(name))
    return match.group(1) if match else str(name)


def block_index_from_name(name: str) -> int:
    """Extract block number from a session folder; return -1 when unavailable."""
    import re

    match = re.search(r"block[-_ ]*(\d+)", str(name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def session_holdout_split(
    trials: list[Trial],
    *,
    test_date: str = "",
) -> tuple[list[Trial], list[Trial], list[Trial], dict]:
    """Hold out one whole session date as the test set.

    This asks whether a decoder trained on earlier sessions transfers to a
    different recording day.  Validation is taken from held-out repetitions of
    the remaining training dates, so the test date is completely unseen.
    """
    dates = sorted({session_date_from_name(t.session) for t in trials})
    if len(dates) < 2:
        raise ValueError("session_holdout needs at least two session dates.")
    chosen_test_date = str(test_date or dates[-1])
    if chosen_test_date not in dates:
        raise ValueError(f"session_holdout test date {chosen_test_date} not found in {dates}.")
    pool = [t for t in trials if session_date_from_name(t.session) != chosen_test_date]
    test = [t for t in trials if session_date_from_name(t.session) == chosen_test_date]
    train = [t for t in pool if int(t.repeat) <= 7]
    val = [t for t in pool if int(t.repeat) >= 8]
    if not train or not val or not test:
        raise ValueError("session_holdout produced an empty train/val/test split.")
    return train, val, test, {
        "train_dates": sorted({session_date_from_name(t.session) for t in train}),
        "val_dates": sorted({session_date_from_name(t.session) for t in val}),
        "test_dates": [chosen_test_date],
        "validation_rule": "repeats >= 8 from non-test dates",
    }


def block_holdout_split(
    trials: list[Trial],
    *,
    test_block: int = 3,
) -> tuple[list[Trial], list[Trial], list[Trial], dict]:
    """Train on two blocks and test on the held-out block.

    With three blocks per session, the default tests block 3 across every date.
    This is the requested section/block generalization split.
    """
    blocks = sorted({block_index_from_name(t.session) for t in trials if block_index_from_name(t.session) > 0})
    if len(blocks) < 2:
        raise ValueError("block_holdout needs at least two block indices in the session names.")
    chosen_test_block = int(test_block)
    if chosen_test_block not in blocks:
        raise ValueError(f"block_holdout test block {chosen_test_block} not found in {blocks}.")
    pool = [t for t in trials if block_index_from_name(t.session) != chosen_test_block]
    test = [t for t in trials if block_index_from_name(t.session) == chosen_test_block]
    train = [t for t in pool if int(t.repeat) <= 7]
    val = [t for t in pool if int(t.repeat) >= 8]
    if not train or not val or not test:
        raise ValueError("block_holdout produced an empty train/val/test split.")
    return train, val, test, {
        "train_blocks": sorted({block_index_from_name(t.session) for t in train}),
        "val_blocks": sorted({block_index_from_name(t.session) for t in val}),
        "test_blocks": [chosen_test_block],
        "validation_rule": "repeats >= 8 from non-test blocks",
    }


def make_split(
    trials: list[Trial],
    mode: str,
    seed: int,
    *,
    session_holdout_test_date: str = "",
    block_holdout_test_block: int = 3,
) -> tuple[list[Trial], list[Trial], list[Trial], dict]:
    """Dispatch between all supported train/validation/test split definitions."""
    if mode == "repeat_holdout":
        # Repeat holdout is the easier split: gestures are present in training,
        # but the last repetitions are held out for validation/test.
        train, val, test = split_trials(trials)
        return train, val, test, {
            "train_repeats": [0, 1, 2, 3, 4, 5],
            "val_repeats": [6, 7],
            "test_repeats": [8, 9],
        }
    if mode == "gesture_holdout":
        return deterministic_gesture_split(trials, seed=seed)
    if mode == "session_holdout":
        return session_holdout_split(trials, test_date=session_holdout_test_date)
    if mode == "block_holdout":
        return block_holdout_split(trials, test_block=block_holdout_test_block)
    raise ValueError(mode)


@dataclass(slots=True)
class NeuralProjector:
    """Training-only PCA transform, or identity transform when PCA is disabled."""

    mode: str
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    explained_variance: float = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the stored neural projection to one neural feature array."""
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "none":
            return x
        if self.mean is None or self.components is None:
            raise ValueError("PCA projector is not fitted.")
        return ((x - self.mean) @ self.components.T).astype(np.float32)


@dataclass(slots=True)
class TargetProjector:
    """Optional PCA transform for the kinematic target.

    `mode="qpos"` means the model predicts all 32 target dimensions directly.
    `mode="qpos_pca"` means the model predicts low-dimensional kinematic PC
    scores; those scores are reconstructed back to 32-D qpos/residual before
    metrics and MuJoCo export.  This keeps the final decoder output unchanged.
    """

    mode: str
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    explained_variance: float = 1.0

    @property
    def output_dim(self) -> int:
        """Number of values the model head predicts before reconstruction."""
        if self.mode == "qpos":
            return N_QPOS
        if self.components is None:
            raise ValueError("qpos PCA target projector has not been fitted.")
        return int(self.components.shape[0])

    def encode(self, y: np.ndarray) -> np.ndarray:
        """Convert 32-D qpos/residual targets into the trained target space."""
        y = np.asarray(y, dtype=np.float32)
        if self.mode == "qpos":
            return y
        if self.mean is None or self.components is None:
            raise ValueError("qpos PCA target projector has not been fitted.")
        return ((y - self.mean) @ self.components.T).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct full 32-D qpos/residual values from model outputs."""
        z = np.asarray(z, dtype=np.float32)
        if self.mode == "qpos":
            return z
        if self.mean is None or self.components is None:
            raise ValueError("qpos PCA target projector has not been fitted.")
        return (z @ self.components + self.mean).astype(np.float32)


def fit_neural_projector(
    train: list[Trial],
    *,
    mode: str,
    components: int,
    seed: int,
    max_rows: int = 120_000,
) -> NeuralProjector:
    """Fit PCA on training neural bins only, or return identity projection."""
    if mode == "none":
        # "No PCA" means the models see all causal neural features directly.
        # In the current data this is usually the 256-channel neural vector after
        # the causal EMA smoothing below.
        return NeuralProjector(mode="none")
    # PCA is fit only on training neural bins.  Validation/test neural bins are
    # projected using this frozen basis, which avoids leakage from the held-out
    # data into preprocessing.
    rows = np.concatenate([causal_ema(t.neural, 0.15) for t in train], axis=0).astype(np.float32)
    if rows.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        rows = rows[rng.choice(rows.shape[0], size=max_rows, replace=False)]
    n_components = min(int(components), rows.shape[1], rows.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(rows)
    return NeuralProjector(
        mode=f"pca{n_components}",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def save_neural_pca_diagnostics(
    train: list[Trial],
    *,
    split_mode: str,
    out_dir: Path,
    marker_components: int,
    seed: int,
    max_rows: int = 120_000,
) -> None:
    """Save a full neural PCA scree/cumulative-variance plot for one split.

    This diagnostic is different from training a pca64 model.  It fits PCA on
    training neural bins only and keeps all possible components, so the plot can
    answer: "how many neural components would be enough for this split?"
    """
    rows = np.concatenate([causal_ema(t.neural, 0.15) for t in train], axis=0).astype(np.float32)
    if rows.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        rows = rows[rng.choice(rows.shape[0], size=max_rows, replace=False)]
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
            "component": np.arange(1, n_components + 1, dtype=np.int64),
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance": cumulative,
        }
    )
    table.to_csv(diag_dir / f"{split_mode}_neural_pca_explained_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["component"], 100.0 * table["cumulative_explained_variance"], color="#2f80ed", linewidth=2.0)
    ax.bar(table["component"], 100.0 * table["explained_variance_ratio"], color="#9ec5fe", alpha=0.45, width=1.0)
    mark = min(int(marker_components), n_components)
    kept = float(cumulative[mark - 1]) if mark >= 1 else float("nan")
    ax.axvline(mark, color="#111111", linestyle="--", linewidth=1.0)
    ax.text(mark, min(96.0, 100.0 * kept + 3.0), f"{mark} PCs = {100.0 * kept:.1f}%", ha="center", va="bottom")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n_components)
    ax.set_xlabel("Neural principal component")
    ax.set_ylabel("Training neural variance explained (%)")
    ax.set_title(f"Neural PCA explained variance: {split_mode}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diag_dir / f"{split_mode}_neural_pca_explained_variance.png", dpi=180)
    plt.close(fig)


@dataclass(slots=True)
class TemplateBank:
    """Mean qpos template trajectories used by hybrid residual decoders."""

    key_name: str
    templates: dict[int, np.ndarray]
    global_template: np.ndarray
    grid_size: int = 180

    def predict(self, key: int | None, n_rows: int) -> np.ndarray:
        """Return the template for one object/key, resampled to a trial length."""
        template = self.templates.get(int(key), self.global_template) if key is not None else self.global_template
        return resample_rows(template, n_rows)


def build_template_bank(train: list[Trial], key_name: str, grid_size: int = 180) -> TemplateBank:
    """Build global and per-key average qpos priors from training trials."""
    # Hybrid decoders use a mean qpos template/prior per object ID.  The model
    # learns the residual around this prior, so it should be described as
    # task-assisted rather than a fully label-free decoder.
    global_stack = np.stack([resample_rows(t.qpos, grid_size) for t in train], axis=0)
    global_template = global_stack.mean(axis=0).astype(np.float32)
    keys = sorted({int(getattr(t, key_name)) for t in train if getattr(t, key_name) is not None and int(getattr(t, key_name)) >= 0})
    templates: dict[int, np.ndarray] = {}
    for key in keys:
        chosen = [t for t in train if getattr(t, key_name) is not None and int(getattr(t, key_name)) == key]
        templates[key] = np.stack([resample_rows(t.qpos, grid_size) for t in chosen], axis=0).mean(axis=0).astype(np.float32)
    return TemplateBank(key_name=key_name, templates=templates, global_template=global_template, grid_size=grid_size)


def raw_targets_for_trials(
    trials: list[Trial],
    *,
    model_mode: str,
    template_bank: TemplateBank | None,
    template_key_name: str,
) -> list[np.ndarray]:
    """Return the 32-D supervised target before optional target PCA.

    Direct models learn qpos itself.  Hybrid models learn the residual
    `qpos - template`, because the template is added back after prediction.
    """
    targets = []
    for trial in trials:
        if model_mode == "direct":
            targets.append(trial.qpos.astype(np.float32))
        elif model_mode == "hybrid":
            assert template_bank is not None
            prior_key = getattr(trial, template_key_name)
            prior_key = int(prior_key) if prior_key is not None else None
            prior = template_bank.predict(prior_key, trial.n_time)
            targets.append((trial.qpos - prior).astype(np.float32))
        else:
            raise ValueError(model_mode)
    return targets


def fit_target_projector(
    train: list[Trial],
    *,
    target_mode: str,
    model_mode: str,
    template_bank: TemplateBank | None,
    template_key_name: str,
    components: int,
    seed: int,
) -> TargetProjector:
    """Fit optional PCA on training kinematic targets only."""
    if target_mode == "qpos":
        return TargetProjector(mode="qpos")
    if target_mode != "qpos_pca":
        raise ValueError(target_mode)
    y_train = np.concatenate(
        raw_targets_for_trials(
            train,
            model_mode=model_mode,
            template_bank=template_bank,
            template_key_name=template_key_name,
        ),
        axis=0,
    ).astype(np.float32)
    n_components = min(int(components), y_train.shape[1], y_train.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(y_train)
    return TargetProjector(
        mode=f"qpos_pca{n_components}",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def object_onehot(trial: Trial, n_objects: int) -> np.ndarray:
    """Create a repeated one-hot object/task label for every time bin in a trial."""
    # This is where object-conditioned models receive object information.  The
    # label is not inferred by this model from the spikes; it is supplied from
    # the session metadata/data loader as trial.object_id, then repeated across
    # all time bins of that trial.
    out = np.zeros((trial.n_time, n_objects), dtype=np.float32)
    if 0 <= int(trial.object_id) < n_objects:
        out[:, int(trial.object_id)] = 1.0
    return out


def neural_features(trial: Trial, projector: NeuralProjector, *, smooth_alpha: float = 0.15) -> np.ndarray:
    """Smooth neural activity causally and apply the selected PCA/identity transform."""
    # The neural input is made online-like here: each bin can only depend on
    # present and past neural activity.  causal_ema is an exponential smoother,
    # and projector is either identity ("none") or the frozen PCA basis.
    return projector.transform(causal_ema(trial.neural, smooth_alpha)).astype(np.float32)


def phase_features(trial: Trial) -> np.ndarray:
    """Create elapsed phase from 0 to 1 across one trial."""
    # Phase is a simple progress signal: 0 near the beginning of a trial and 1
    # near the end.  It helps models know where they are in a stereotyped reach
    # and grasp attempt, but it is also a task-assistance signal that would need
    # to be generated online by a clock/attempt-state machine.
    return np.linspace(0.0, 1.0, trial.n_time, dtype=np.float32)[:, None]


def direct_features(trial: Trial, projector: NeuralProjector, n_objects: int, condition_object: bool) -> np.ndarray:
    """Build inputs for direct decoders that predict full qpos directly."""
    # Direct mode: neural history + phase, optionally with an object/task label,
    # is asked to predict the full qpos trajectory.
    parts = [neural_features(trial, projector), phase_features(trial)]
    if condition_object:
        parts.append(object_onehot(trial, n_objects))
    return np.concatenate(parts, axis=1).astype(np.float32)


def hybrid_features(trial: Trial, projector: NeuralProjector, prior: np.ndarray, n_objects: int) -> np.ndarray:
    """Build inputs for hybrid decoders that predict qpos residuals around a prior."""
    # Hybrid mode: the input contains the neural features, elapsed phase, a
    # template/prior trajectory, and the object one-hot vector.  The target is
    # qpos - prior, so the final prediction becomes prior + residual.
    return np.concatenate(
        [neural_features(trial, projector), phase_features(trial), prior.astype(np.float32), object_onehot(trial, n_objects)],
        axis=1,
    ).astype(np.float32)


def concat_xy(
    trials: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    model_mode: str,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    template_key_name: str = "object_id",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Concatenate per-trial features/targets into matrices for linear training."""
    # Ridge is trained on one big 2-D matrix:
    #   X = all time bins from all training trials stacked together
    #   Y = the matching qpos targets, or qpos residuals for hybrid mode
    # Sequence models keep the trials separate, but they use the same feature
    # construction rules in sequence_items().
    xs, ys, priors = [], [], []
    for trial in trials:
        if model_mode == "direct":
            prior = np.zeros_like(trial.qpos, dtype=np.float32)
            x = direct_features(trial, projector, n_objects, condition_object)
            y_raw = trial.qpos.astype(np.float32)
        elif model_mode == "hybrid":
            assert template_bank is not None
            prior_key = getattr(trial, template_key_name)
            prior_key = int(prior_key) if prior_key is not None else None
            prior = template_bank.predict(prior_key, trial.n_time)
            x = hybrid_features(trial, projector, prior, n_objects)
            y_raw = (trial.qpos - prior).astype(np.float32)
        else:
            raise ValueError(model_mode)
        xs.append(x)
        ys.append(target_projector.encode(y_raw))
        priors.append(prior)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), priors


def eval_metric_extras(y_true: np.ndarray, y_pred: np.ndarray, *, sample_rate_hz: float) -> dict[str, float]:
    """Compute the base metrics plus trajectory-shape and timing diagnostics."""
    base = summarize(y_true, y_pred)
    extras = {
        "procrustes_corr": procrustes_corr(y_true, y_pred),
        "procrustes_disparity": procrustes_disparity(y_true, y_pred),
        "onset_delay_s": onset_offset_delay(y_true, y_pred, sample_rate_hz=sample_rate_hz)[0],
        "offset_delay_s": onset_offset_delay(y_true, y_pred, sample_rate_hz=sample_rate_hz)[1],
        "best_lag_bins": lag_sensitivity(y_true, y_pred)[0],
        "best_lag_R2": lag_sensitivity(y_true, y_pred)[1],
    }
    return {**base, **extras}


def aggregate_trial_metrics(trials: list[Trial], preds: list[np.ndarray], *, sample_rate_hz: float) -> dict[str, float]:
    """Aggregate test/validation metrics over a list of trial predictions."""
    y_true = np.concatenate([t.qpos[: p.shape[0]] for t, p in zip(trials, preds, strict=True)], axis=0)
    y_pred = np.concatenate(preds, axis=0)
    base = summarize(y_true, y_pred)
    onset = []
    offset = []
    for trial, pred in zip(trials, preds, strict=True):
        n = min(trial.qpos.shape[0], pred.shape[0])
        d0, d1 = onset_offset_delay(trial.qpos[:n], pred[:n], sample_rate_hz=sample_rate_hz)
        onset.append(d0)
        offset.append(d1)
    best_lag, best_lag_r2 = lag_sensitivity(y_true, y_pred)
    return {
        **base,
        "procrustes_corr": procrustes_corr(y_true, y_pred),
        "procrustes_disparity": procrustes_disparity(y_true, y_pred),
        "onset_delay_s": float(np.mean(onset)) if onset else float("nan"),
        "onset_delay_std_s": float(np.std(onset)) if onset else float("nan"),
        "offset_delay_s": float(np.mean(offset)) if offset else float("nan"),
        "offset_delay_std_s": float(np.std(offset)) if offset else float("nan"),
        "best_lag_bins": best_lag,
        "best_lag_R2": best_lag_r2,
    }


def procrustes_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure global trajectory-shape similarity after centering and scaling."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an <= 1e-12 or bn <= 1e-12:
        return float("nan")
    return float(np.sum((a / an) * (b / bn)))


def procrustes_disparity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Convert Procrustes correlation into a lower-is-better mismatch score."""
    corr = procrustes_corr(y_true, y_pred)
    return float(2.0 * (1.0 - corr)) if math.isfinite(corr) else float("nan")


def movement_energy(y: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame movement magnitude used for onset/offset estimates."""
    if y.shape[0] < 2:
        return np.zeros(y.shape[0], dtype=np.float32)
    vel = np.diff(y, axis=0, prepend=y[:1])
    return np.linalg.norm(vel, axis=1).astype(np.float32)


def onset_offset_delay(y_true: np.ndarray, y_pred: np.ndarray, *, sample_rate_hz: float) -> tuple[float, float]:
    """Estimate decoded movement start/end timing error in seconds."""
    def bounds(y: np.ndarray) -> tuple[int, int]:
        """Find active movement bounds from a relative movement-energy threshold."""
        e = movement_energy(y)
        if np.max(e) <= 1e-12:
            return 0, max(0, len(e) - 1)
        active = np.where(e >= 0.2 * np.max(e))[0]
        if active.size == 0:
            return 0, max(0, len(e) - 1)
        return int(active[0]), int(active[-1])

    t0, t1 = bounds(y_true)
    p0, p1 = bounds(y_pred)
    return (p0 - t0) / sample_rate_hz, (p1 - t1) / sample_rate_hz


def lag_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 5) -> tuple[int, float]:
    """Find the small temporal shift that would maximize R2 as a diagnostic."""
    best_lag = 0
    best_r2 = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            yt, yp = y_true[-lag:], y_pred[:lag]
        elif lag > 0:
            yt, yp = y_true[:-lag], y_pred[lag:]
        else:
            yt, yp = y_true, y_pred
        if yt.shape[0] < 3:
            continue
        r2 = summarize(yt, yp)["R2"]
        if math.isfinite(r2) and r2 > best_r2:
            best_r2 = float(r2)
            best_lag = lag
    return best_lag, best_r2


@dataclass(slots=True)
class RidgeDecoder:
    """Saved pieces of one ridge model: scalers, weights, and mode metadata."""

    x_scaler: Standardizer
    y_scaler: Standardizer
    weights: np.ndarray
    bias: np.ndarray
    model_mode: str
    condition_object: bool
    model_name: str
    target_projector: TargetProjector
    template_key_name: str = "object_id"

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        """Predict in standardized target space."""
        return self.x_scaler.transform(x).astype(np.float32) @ self.weights + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict qpos/residual values in original physical units."""
        return self.y_scaler.inverse(self.predict_scaled(x)).astype(np.float32)


def train_ridge_decoder(
    train: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    model_mode: str,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    alpha: float,
    template_key_name: str = "object_id",
) -> tuple[RidgeDecoder, float]:
    """Fit one ridge decoder and return the model plus training time in ms."""
    # Ridge is the easiest baseline to explain: after feature construction and
    # standardization, it solves one regularized linear equation from X to Y.
    # There is no recurrence and no hidden temporal state here.
    x_train, y_train, _ = concat_xy(
        train,
        projector,
        n_objects,
        model_mode=model_mode,
        template_bank=template_bank,
        condition_object=condition_object,
        target_projector=target_projector,
        template_key_name=template_key_name,
    )
    t0 = time.perf_counter()
    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)
    weights, bias = _fit_ridge_readout(
        x_scaler.transform(x_train).astype(np.float32),
        y_scaler.transform(y_train).astype(np.float32),
        alpha=alpha,
    )
    return (
        RidgeDecoder(
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            weights=weights,
            bias=bias,
            model_mode=model_mode,
            condition_object=condition_object,
            model_name=(
                f"{model_mode}_ridge{'_object_conditioned' if condition_object else ''}_{template_key_name}_template"
                if model_mode == "hybrid"
                else f"{model_mode}_ridge{'_object_conditioned' if condition_object else ''}"
            ),
            target_projector=target_projector,
            template_key_name=template_key_name,
        ),
        (time.perf_counter() - t0) * 1000.0,
    )


def predict_ridge_trials(
    model: RidgeDecoder,
    trials: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    template_bank: TemplateBank | None,
    output_alpha: float,
) -> list[np.ndarray]:
    """Run one trained ridge decoder on trials and optionally smooth the output."""
    preds = []
    for trial in trials:
        if model.model_mode == "direct":
            x = direct_features(trial, projector, n_objects, model.condition_object)
            pred = model.target_projector.decode(model.predict(x))
        else:
            assert template_bank is not None
            prior_key = getattr(trial, model.template_key_name)
            prior = template_bank.predict(int(prior_key) if prior_key is not None else None, trial.n_time)
            x = hybrid_features(trial, projector, prior, n_objects)
            pred = prior + model.target_projector.decode(model.predict(x))
        preds.append(smooth_output(pred.astype(np.float32), alpha=output_alpha).astype(np.float32))
    return preds


def fit_kalman_matrices(states: list[np.ndarray], obs: list[np.ndarray], ridge: float = 1e-3) -> dict:
    """Fit linear state-transition and observation matrices for a Kalman filter."""
    # Kalman uses a state-space view.  The hidden state z is qpos+qvel, and the
    # observation x is the neural feature vector.  Training estimates:
    #   A: how z evolves from one bin to the next
    #   H: how neural observations relate to z
    #   Q/R: simple diagonal process/observation noise levels
    z_prev = np.concatenate([z[:-1] for z in states], axis=0)
    z_next = np.concatenate([z[1:] for z in states], axis=0)
    x_now = np.concatenate(obs, axis=0)
    z_now = np.concatenate(states, axis=0)
    z_mean = z_now.mean(axis=0).astype(np.float32)
    x_mean = x_now.mean(axis=0).astype(np.float32)
    A_w, A_b = _ridge_affine(z_prev, z_next, ridge)
    H_w, H_b = _ridge_affine(z_now, x_now, ridge)
    trans_res = z_next - (z_prev @ A_w + A_b)
    obs_res = x_now - (z_now @ H_w + H_b)
    q = np.var(trans_res, axis=0).astype(np.float32) + 1e-4
    r = np.var(obs_res, axis=0).astype(np.float32) + 1e-3
    return {"A": A_w.T.astype(np.float32), "A_b": A_b, "H": H_w.T.astype(np.float32), "H_b": H_b, "Q_diag": q, "R_diag": r, "z_mean": z_mean, "x_mean": x_mean}


def _ridge_affine(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve y ~= x @ W + b with ridge regularization."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    gram = xb.T @ xb
    reg = np.eye(gram.shape[0], dtype=np.float32) * np.float32(ridge)
    reg[-1, -1] = 0.0
    wb = np.linalg.solve(gram + reg, xb.T @ y)
    return wb[:-1].astype(np.float32), wb[-1].astype(np.float32)


def kalman_filter_trial(params: dict, obs: np.ndarray) -> np.ndarray:
    """Run one causal Kalman filter pass over a single neural-observation trial."""
    A = params["A"]
    H = params["H"]
    A_b = params["A_b"]
    H_b = params["H_b"]
    q = params["Q_diag"]
    r = params["R_diag"]
    state_dim = A.shape[0]
    z = params["z_mean"].astype(np.float32).copy()
    p = np.eye(state_dim, dtype=np.float32)
    Q = np.diag(q).astype(np.float32)
    R = np.diag(r).astype(np.float32)
    out = np.empty((obs.shape[0], state_dim), dtype=np.float32)
    for t in range(obs.shape[0]):
        z_pred = A @ z + A_b
        p_pred = A @ p @ A.T + Q
        innovation = obs[t] - (H @ z_pred + H_b)
        S = H @ p_pred @ H.T + R
        try:
            K = p_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = p_pred @ H.T @ np.linalg.pinv(S)
        z = z_pred + K @ innovation
        p = (np.eye(state_dim, dtype=np.float32) - K @ H) @ p_pred
        out[t] = z
    return out


def corrected_pv_state(trial: Trial, refit: bool) -> np.ndarray:
    """Build qpos+qvel state, optionally rotating velocity toward the final target."""
    # Classical ReFIT BCIs often replace noisy/meandering velocity labels with
    # a velocity pointing toward the intended target.  This implementation keeps
    # qpos unchanged and only rotates the qvel part toward the final qpos of the
    # trial, as a simple offline ReFIT-style approximation.
    qvel = trial.qvel.astype(np.float32)
    if refit:
        goal = trial.qpos[-1]
        direction = goal[None, :] - trial.qpos
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        unit = direction / np.maximum(norm, 1e-6)
        speed = np.linalg.norm(qvel, axis=1, keepdims=True)
        qvel = (unit * speed).astype(np.float32)
    return np.concatenate([trial.qpos.astype(np.float32), qvel], axis=1)


def train_eval_kalman(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector: NeuralProjector,
    *,
    refit: bool,
    sample_rate_hz: float,
) -> tuple[dict, list[np.ndarray], float]:
    """Train/evaluate classical Kalman or ReFIT-style Kalman on one split."""
    states = [corrected_pv_state(t, refit=refit) for t in train]
    obs = [neural_features(t, projector) for t in train]
    t0 = time.perf_counter()
    params = fit_kalman_matrices(states, obs, ridge=1e-2)
    train_ms = (time.perf_counter() - t0) * 1000.0

    def predict(split: list[Trial]) -> list[np.ndarray]:
        """Decode qpos for a validation/test split using the fitted Kalman parameters."""
        out = []
        for trial in split:
            z = kalman_filter_trial(params, neural_features(trial, projector))
            out.append(z[:, :N_QPOS].astype(np.float32))
        return out

    val_preds = predict(val)
    test_preds = predict(test)
    offline_latency = latency_ms_per_bin(lambda: predict(test), total_bins=sum(t.n_time for t in test))
    row = {
        "model_family": "refit_kalman" if refit else "kalman",
        "model_mode": "direct_pv_state",
        "condition_object": False,
        "output_alpha": 1.0,
        "train_ms": train_ms,
        "latency_ms_per_bin_offline": offline_latency,
        **{f"val_{k}": v for k, v in aggregate_trial_metrics(val, val_preds, sample_rate_hz=sample_rate_hz).items()},
        **{f"test_{k}": v for k, v in aggregate_trial_metrics(test, test_preds, sample_rate_hz=sample_rate_hz).items()},
    }
    return row, test_preds, train_ms


class SequenceNetFactory:
    """Factory for the GRU, LFADS-lite, and Transformer sequence architectures."""

    @staticmethod
    def make(
        torch,
        nn,
        model_family: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        factor_dim: int,
        bidirectional_lfads: bool,
        gru_layers: int = 1,
    ):
        """Create one torch model with a shared predict qpos/residual interface."""
        # All sequence nets receive a padded batch shaped
        #   (batch, time, input_dim)
        # and return one output vector for every valid time bin.  The output is
        # either 32-D qpos/residual or low-D qpos-PC scores. The mask tells
        # the loss which entries are real data and which are padding.
        class GRUNet(nn.Module):
            """Causal GRU decoder: recurrent hidden state -> 32-D qpos/residual."""

            def __init__(self):
                super().__init__()
                # The GRU reads bins in time order.  At time t it has only seen
                # bins <= t, so this variant is compatible with online decoding.
                self.gru = nn.GRU(
                    input_dim,
                    hidden_dim,
                    num_layers=int(gru_layers),
                    batch_first=True,
                    dropout=0.05 if int(gru_layers) > 1 else 0.0,
                )
                self.norm = nn.LayerNorm(hidden_dim)
                # The head maps the recurrent hidden state at each bin into
                # the selected target representation.
                self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

            def forward(self, x, mask):
                """Run the GRU over a padded batch; losses use mask later."""
                h, _ = self.gru(x)
                h = self.norm(h)
                return {"y": self.head(h), "recon": None, "latent": h}

        class LfadsLiteNet(nn.Module):
            """LFADS-lite decoder: recurrent encoder -> low-D factors -> qpos/residual."""

            def __init__(self):
                super().__init__()
                # LFADS-lite here is a compact supervised approximation: an
                # encoder GRU creates a temporal hidden representation, this is
                # compressed into low-dimensional factors, and a readout maps
                # those factors to qpos/residual.
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional_lfads)
                enc_dim = hidden_dim * (2 if bidirectional_lfads else 1)
                self.factor = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, factor_dim), nn.Tanh())
                self.head = nn.Sequential(nn.Linear(factor_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
                # The reconstruction head asks factors to also reconstruct the
                # input features.  This is a light LFADS-like auxiliary task
                # that encourages factors to preserve neural dynamics.
                self.recon = nn.Linear(factor_dim, input_dim)

            def forward(self, x, mask):
                """Return decoded output plus neural-feature reconstruction."""
                h, _ = self.gru(x)
                f = self.factor(h)
                return {"y": self.head(f), "recon": self.recon(f), "latent": f}

        class CausalTransformerNet(nn.Module):
            """Causal Transformer decoder with masked attention over past/current bins."""

            def __init__(self):
                super().__init__()
                # The Transformer projects each time bin to hidden_dim, then
                # uses causal self-attention so each bin can attend to previous
                # bins but not future bins.
                self.in_proj = nn.Linear(input_dim, hidden_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                    dropout=0.08,
                    activation="gelu",
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

            def forward(self, x, mask):
                """Apply causal self-attention and predict qpos/residual per bin."""
                h = self.in_proj(x)
                n = h.shape[1]
                causal_mask = torch.triu(torch.ones(n, n, device=h.device, dtype=torch.bool), diagonal=1)
                h = self.encoder(h, mask=causal_mask, src_key_padding_mask=~mask)
                return {"y": self.head(h), "recon": None, "latent": h}

        if model_family == "gru":
            return GRUNet()
        if model_family == "lfads_lite":
            return LfadsLiteNet()
        if model_family == "transformer":
            return CausalTransformerNet()
        raise ValueError(model_family)


def sequence_items(
    trials: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    model_mode: str,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    template_key_name: str,
    x_scaler: Standardizer | None,
    y_scaler: Standardizer | None,
) -> tuple[list[dict], Standardizer, Standardizer]:
    """Prepare variable-length sequence examples for GRU/LFADS/Transformer."""
    # This is the sequence-model equivalent of concat_xy().
    # Each item keeps one complete trial, because recurrent/attention models
    # need the temporal order.  Standardizers are fit only on training items and
    # reused for validation/test.
    raw = []
    xs, ys = [], []
    for trial in trials:
        if model_mode == "direct":
            prior = np.zeros_like(trial.qpos, dtype=np.float32)
            x = direct_features(trial, projector, n_objects, condition_object)
            y_raw = trial.qpos.astype(np.float32)
        else:
            assert template_bank is not None
            prior_key = getattr(trial, template_key_name)
            prior_key = int(prior_key) if prior_key is not None else None
            prior = template_bank.predict(prior_key, trial.n_time)
            x = hybrid_features(trial, projector, prior, n_objects)
            y_raw = (trial.qpos - prior).astype(np.float32)
        y = target_projector.encode(y_raw)
        raw.append({"trial": trial, "x": x, "y": y, "prior": prior, "true": trial.qpos.astype(np.float32)})
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


def iter_batches(items: list[dict], batch_size: int, rng: np.random.Generator, shuffle: bool):
    """Yield minibatches of trial dictionaries, optionally shuffled each epoch."""
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [items[i] for i in idx[start : start + batch_size]]


def pad_batch(batch: list[dict], torch, device):
    """Pad variable-length trials to one tensor and create a valid-time mask."""
    lengths = np.asarray([b["x"].shape[0] for b in batch], dtype=np.int64)
    max_len = int(lengths.max())
    input_dim = batch[0]["x"].shape[1]
    output_dim = batch[0]["y"].shape[1]
    xb = np.zeros((len(batch), max_len, input_dim), dtype=np.float32)
    yb = np.zeros((len(batch), max_len, output_dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=bool)
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        xb[i, :n] = item["x"]
        yb[i, :n] = item["y"]
        mask[i, :n] = True
    return torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device), torch.from_numpy(mask).to(device)


def masked_mse(pred, target, mask):
    """Mean squared error over valid, non-padding time bins only."""
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target) ** 2 * mask_f).sum() / (mask_f.sum() * pred.shape[-1]).clamp_min(1.0)


def masked_delta_mse(pred, target, mask):
    """Velocity-like loss term that encourages correct frame-to-frame changes."""
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(pred.dtype)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return ((dp - dt) ** 2 * valid).sum() / (valid.sum() * pred.shape[-1]).clamp_min(1.0)


def eval_seq_loss(model, items, torch, device, batch_size, recon_weight):
    """Evaluate neural-model validation loss without updating weights."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in iter_batches(items, batch_size, np.random.default_rng(0), shuffle=False):
            xb, yb, mask = pad_batch(batch, torch, device)
            out = model(xb, mask)
            loss = masked_mse(out["y"], yb, mask) + 0.05 * masked_delta_mse(out["y"], yb, mask)
            if out["recon"] is not None:
                loss = loss + recon_weight * masked_mse(out["recon"], xb, mask)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_seq(
    model,
    items,
    y_scaler: Standardizer,
    target_projector: TargetProjector,
    torch,
    device,
    *,
    model_mode: str,
    output_alpha: float,
) -> list[np.ndarray]:
    """Decode qpos/residual trajectories for each trial from a trained sequence model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for item in items:
            xb = torch.from_numpy(item["x"][None]).to(device)
            mask = torch.ones((1, item["x"].shape[0]), dtype=torch.bool, device=device)
            out = model(xb, mask)["y"][0].detach().cpu().numpy().astype(np.float32)
            y = target_projector.decode(y_scaler.inverse(out).astype(np.float32))
            if model_mode == "hybrid":
                y = item["prior"] + y
            preds.append(smooth_output(y, output_alpha).astype(np.float32))
    return preds


def train_sequence_decoder(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    model_family: str,
    model_mode: str,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    template_key_name: str,
    device_name: str,
    profile: str,
    seed: int,
    bidirectional_lfads: bool,
    gru_layers: int,
    sample_rate_hz: float,
    out_dir: Path,
) -> tuple[dict, list[np.ndarray], pd.DataFrame]:
    """Train one GRU/LFADS-lite/Transformer configuration and select by validation score."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device(device_name)
    train_items, x_scaler, y_scaler = sequence_items(
        train,
        projector,
        n_objects,
        model_mode=model_mode,
        template_bank=template_bank,
        condition_object=condition_object,
        target_projector=target_projector,
        template_key_name=template_key_name,
        x_scaler=None,
        y_scaler=None,
    )
    val_items, _, _ = sequence_items(
        val,
        projector,
        n_objects,
        model_mode=model_mode,
        template_bank=template_bank,
        condition_object=condition_object,
        target_projector=target_projector,
        template_key_name=template_key_name,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    test_items, _, _ = sequence_items(
        test,
        projector,
        n_objects,
        model_mode=model_mode,
        template_bank=template_bank,
        condition_object=condition_object,
        target_projector=target_projector,
        template_key_name=template_key_name,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    if profile == "smoke":
        # Smoke is intentionally tiny: it only checks that loading/training works.
        hidden_dim, factor_dim, epochs, patience, batch_size = 40, 16, 2, 2, 24
    elif profile == "quick":
        # Quick is for fast iteration on a laptop/VM.
        hidden_dim, factor_dim, epochs, patience, batch_size = 56, 24, 16, 5, 32
    else:
        # Full is the thesis-style run used for the main VM results.
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
        # Training loss = qpos/residual error + small delta/velocity-shape error.
        # LFADS-lite additionally receives the reconstruction auxiliary loss.
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
        print(f"{model_family}/{model_mode}/obj={condition_object} epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val - 1e-4:
            # Best epoch is selected by validation loss, not test performance.
            # The test split stays untouched until the selected model is scored.
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_ms = (time.perf_counter() - t0) * 1000.0
    rows = []
    best_row = None
    best_preds = None
    for output_alpha in (1.0, 0.75, 0.55, 0.35):
        # output_alpha is a post-decoding smoother.  It does not retrain the
        # model; it checks whether a smoother command would be better for online
        # visualization/control.
        val_preds = predict_seq(model, val_items, y_scaler, target_projector, torch, device, model_mode=model_mode, output_alpha=output_alpha)
        test_preds = predict_seq(model, test_items, y_scaler, target_projector, torch, device, model_mode=model_mode, output_alpha=output_alpha)
        val_trials = [item["trial"] for item in val_items]
        test_trials = [item["trial"] for item in test_items]
        row = {
            "model_family": model_family,
            "model_mode": model_mode,
            "condition_object": condition_object,
            "target_mode": target_projector.mode,
            "target_pca_components": target_projector.output_dim if target_projector.mode != "qpos" else np.nan,
            "target_pca_explained_variance": target_projector.explained_variance if target_projector.mode != "qpos" else np.nan,
            "output_alpha": output_alpha,
            "hidden_dim": hidden_dim,
            "factor_dim": factor_dim if model_family == "lfads_lite" else np.nan,
            "bidirectional": bool(bidirectional_lfads and model_family == "lfads_lite"),
            "gru_layers": int(gru_layers) if model_family == "gru" else np.nan,
            "best_epoch": best_epoch,
            "train_ms": train_ms,
            **{f"val_{k}": v for k, v in aggregate_trial_metrics(val_trials, val_preds, sample_rate_hz=sample_rate_hz).items()},
            **{f"test_{k}": v for k, v in aggregate_trial_metrics(test_trials, test_preds, sample_rate_hz=sample_rate_hz).items()},
        }
        score = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
        row["selection_score"] = score
        rows.append(row)
        if best_row is None or score > best_row["selection_score"]:
            best_row = row
            best_preds = test_preds
    artifact = out_dir / f"{model_family}_{model_mode}_{'obj' if condition_object else 'nolabel'}.pt"
    torch.save(
        {
            # The saved .pt contains everything needed to reproduce inference:
            # preprocessing projector, standardizers, architecture sizes, and
            # the PyTorch weights from the best validation epoch.
            "model_family": model_family,
            "model_mode": model_mode,
            "condition_object": condition_object,
            "target_mode": target_projector.mode,
            "target_mean": target_projector.mean,
            "target_components": target_projector.components,
            "target_explained_variance": target_projector.explained_variance,
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
            "output_dim": target_projector.output_dim,
            "state_dict": best_state,
        },
        artifact,
    )
    assert best_row is not None and best_preds is not None
    best_row["latency_ms_per_bin_offline"] = latency_ms_per_bin(
        lambda: predict_seq(
            model,
            test_items,
            y_scaler,
            target_projector,
            torch,
            device,
            model_mode=model_mode,
            output_alpha=float(best_row["output_alpha"]),
        ),
        total_bins=sum(item["trial"].n_time for item in test_items),
    )
    best_row = {**best_row, "artifact": str(artifact)}
    history_df = pd.DataFrame(history)
    history_df["model_family"] = model_family
    history_df["model_mode"] = model_mode
    history_df["condition_object"] = condition_object
    history_df["target_mode"] = target_projector.mode
    return best_row, best_preds, history_df


def train_eval_ridge_family(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    model_mode: str,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    sample_rate_hz: float,
    template_key_name: str = "object_id",
) -> tuple[dict, list[np.ndarray]]:
    """Train/evaluate one ridge configuration and pick the best output smoothing."""
    model, train_ms = train_ridge_decoder(
        train,
        projector,
        n_objects,
        model_mode=model_mode,
        template_bank=template_bank,
        condition_object=condition_object,
        target_projector=target_projector,
        alpha=100.0,
        template_key_name=template_key_name,
    )
    best_row = None
    best_preds = None
    for output_alpha in (1.0, 0.75, 0.55, 0.35):
        val_preds = predict_ridge_trials(model, val, projector, n_objects, template_bank=template_bank, output_alpha=output_alpha)
        test_preds = predict_ridge_trials(model, test, projector, n_objects, template_bank=template_bank, output_alpha=output_alpha)
        row = {
            "model_family": "ridge",
            "model_mode": model_mode,
            "condition_object": condition_object,
            "template_key_name": template_key_name,
            "target_mode": target_projector.mode,
            "target_pca_components": target_projector.output_dim if target_projector.mode != "qpos" else np.nan,
            "target_pca_explained_variance": target_projector.explained_variance if target_projector.mode != "qpos" else np.nan,
            "output_alpha": output_alpha,
            "train_ms": train_ms,
            **{f"val_{k}": v for k, v in aggregate_trial_metrics(val, val_preds, sample_rate_hz=sample_rate_hz).items()},
            **{f"test_{k}": v for k, v in aggregate_trial_metrics(test, test_preds, sample_rate_hz=sample_rate_hz).items()},
        }
        row["selection_score"] = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row
            best_preds = test_preds
    assert best_row is not None and best_preds is not None
    best_row["latency_ms_per_bin_offline"] = latency_ms_per_bin(
        lambda: predict_ridge_trials(
            model,
            test,
            projector,
            n_objects,
            template_bank=template_bank,
            output_alpha=float(best_row["output_alpha"]),
        ),
        total_bins=sum(t.n_time for t in test),
    )
    return best_row, best_preds


@dataclass(slots=True)
class QposPcaRidgeDecoder:
    """Ridge decoder that predicts low-D qpos PCs, then reconstructs 32-D qpos."""

    x_scaler: Standardizer
    z_scaler: Standardizer
    weights: np.ndarray
    bias: np.ndarray
    qpos_mean: np.ndarray
    qpos_components: np.ndarray
    condition_object: bool
    qpos_pca_explained_variance: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict qpos by going X -> qpos PCs -> reconstructed 32-D qpos."""
        xs = self.x_scaler.transform(x).astype(np.float32)
        z_scaled = xs @ self.weights + self.bias
        z = self.z_scaler.inverse(z_scaled).astype(np.float32)
        return (z @ self.qpos_components + self.qpos_mean).astype(np.float32)


def train_qpos_pca_ridge_decoder(
    train: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    condition_object: bool,
    qpos_pca_components: int,
    alpha: float,
    seed: int,
) -> tuple[QposPcaRidgeDecoder, float]:
    """Fit Ridge on neural features while compressing the qpos target with PCA."""
    # This experiment applies PCA to the target, not the input.  Ridge predicts
    # qpos principal-component scores, and those scores are reconstructed back
    # into the full 32-D qpos command before metrics or MuJoCo export.
    x_train = np.concatenate(
        [direct_features(t, projector, n_objects, condition_object) for t in train],
        axis=0,
    )
    y_train = np.concatenate([t.qpos.astype(np.float32) for t in train], axis=0)
    n_components = min(int(qpos_pca_components), y_train.shape[1], y_train.shape[0] - 1)
    target_pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    target_pca.fit(y_train)
    z_train = target_pca.transform(y_train).astype(np.float32)

    t0 = time.perf_counter()
    x_scaler = Standardizer.fit(x_train)
    z_scaler = Standardizer.fit(z_train)
    weights, bias = _fit_ridge_readout(
        x_scaler.transform(x_train).astype(np.float32),
        z_scaler.transform(z_train).astype(np.float32),
        alpha=alpha,
    )
    train_ms = (time.perf_counter() - t0) * 1000.0
    return (
        QposPcaRidgeDecoder(
            x_scaler=x_scaler,
            z_scaler=z_scaler,
            weights=weights,
            bias=bias,
            qpos_mean=target_pca.mean_.astype(np.float32),
            qpos_components=target_pca.components_.astype(np.float32),
            condition_object=condition_object,
            qpos_pca_explained_variance=float(np.sum(target_pca.explained_variance_ratio_)),
        ),
        train_ms,
    )


def predict_qpos_pca_ridge_trials(
    model: QposPcaRidgeDecoder,
    trials: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    output_alpha: float,
) -> list[np.ndarray]:
    """Predict and reconstruct 32-D qpos for qpos-PCA Ridge experiments."""
    preds = []
    for trial in trials:
        x = direct_features(trial, projector, n_objects, model.condition_object)
        pred = model.predict(x)
        preds.append(smooth_output(pred.astype(np.float32), alpha=output_alpha).astype(np.float32))
    return preds


def train_eval_qpos_pca_ridge_family(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    *,
    condition_object: bool,
    sample_rate_hz: float,
    qpos_pca_components: int,
    seed: int,
) -> tuple[dict, list[np.ndarray]]:
    """Train/evaluate reduced-target Ridge and pick the best output smoothing."""
    model, train_ms = train_qpos_pca_ridge_decoder(
        train,
        projector,
        n_objects,
        condition_object=condition_object,
        qpos_pca_components=qpos_pca_components,
        alpha=100.0,
        seed=seed,
    )
    best_row = None
    best_preds = None
    for output_alpha in (1.0, 0.75, 0.55, 0.35):
        val_preds = predict_qpos_pca_ridge_trials(model, val, projector, n_objects, output_alpha=output_alpha)
        test_preds = predict_qpos_pca_ridge_trials(model, test, projector, n_objects, output_alpha=output_alpha)
        row = {
            "model_family": "ridge_qpos_pca",
            "model_mode": "direct_qpos_pca",
            "condition_object": condition_object,
            "template_key_name": "",
            "qpos_pca_components": int(model.qpos_components.shape[0]),
            "qpos_pca_explained_variance": model.qpos_pca_explained_variance,
            "output_alpha": output_alpha,
            "train_ms": train_ms,
            **{f"val_{k}": v for k, v in aggregate_trial_metrics(val, val_preds, sample_rate_hz=sample_rate_hz).items()},
            **{f"test_{k}": v for k, v in aggregate_trial_metrics(test, test_preds, sample_rate_hz=sample_rate_hz).items()},
        }
        row["selection_score"] = row["val_R2"] + 0.02 * row["val_CC"] - 0.01 * row["val_jitter"]
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row
            best_preds = test_preds
    assert best_row is not None and best_preds is not None
    best_row["latency_ms_per_bin_offline"] = latency_ms_per_bin(
        lambda: predict_qpos_pca_ridge_trials(
            model,
            test,
            projector,
            n_objects,
            output_alpha=float(best_row["output_alpha"]),
        ),
        total_bins=sum(t.n_time for t in test),
    )
    return best_row, best_preds


def qpos_to_ctrl(qpos: np.ndarray, ctrl_to_qpos_index: list[int | None]) -> np.ndarray:
    """Convert predicted qpos order to MuJoCo ctrl order for visualization files."""
    out = np.zeros((qpos.shape[0], len(ctrl_to_qpos_index)), dtype=np.float32)
    for ctrl_i, qpos_i in enumerate(ctrl_to_qpos_index):
        if qpos_i is not None:
            out[:, ctrl_i] = qpos[:, qpos_i]
    return out


def export_mujoco_compare(test: list[Trial], preds: list[np.ndarray], out_dir: Path, model_label: str, max_trials: int = 24) -> None:
    """Save actual-vs-decoded ctrl arrays that the MuJoCo viewer can replay."""
    metadata = parse_mujoco_metadata(DEFAULT_MUJOCO_XML)
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for i, (trial, pred) in enumerate(zip(test, preds, strict=True)):
        if i >= max_trials:
            break
        decoded_ctrl = qpos_to_ctrl(pred, metadata.ctrl_to_qpos_index)
        n = min(decoded_ctrl.shape[0], trial.ctrl.shape[0])
        stem = f"gesture_{trial.gesture:03d}_repeat_{trial.repeat:02d}"
        actual_path = out_dir / f"{stem}_actual_ctrl.npy"
        decoded_path = out_dir / f"{stem}_decoded_ctrl.npy"
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
                "model": model_label,
                "decoded_output": "qpos predicted by model, converted to MuJoCo ctrl order only for visualization",
                "exports": exports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def latency_ms_per_bin(predict_fn, total_bins: int, repeats: int = 1) -> float:
    """Estimate inference wall time normalized per decoded bin."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times) / max(1, int(total_bins)))


def policy_label(row_or_obj) -> str:
    """Return a short human label for direct/object/hybrid conditions."""
    getter = row_or_obj.get if isinstance(row_or_obj, dict) else lambda key, default=None: getattr(row_or_obj, key, default)
    family = str(getter("model_family", ""))
    mode = str(getter("model_mode", ""))
    condition_object = bool(getter("condition_object", False))
    template_key = str(getter("template_key_name", ""))
    if "handle_known" in str(getter("experiment_id", "")) or template_key == "handle_id":
        return "handle-known template"
    if mode == "hybrid":
        return "hybrid object template"
    if condition_object:
        return "direct + object"
    if family in {"kalman", "refit_kalman"}:
        return "direct no label"
    return "direct no label"


def pca_condition_label(row_or_obj) -> str:
    """Return a compact label describing neural PCA and kinematic target PCA."""
    getter = row_or_obj.get if isinstance(row_or_obj, dict) else lambda key, default=None: getattr(row_or_obj, key, default)
    pca_mode = str(getter("pca_mode", "none"))
    target_mode = str(getter("target_mode", "qpos"))
    neural = "neural PCA" if pca_mode != "none" else "no neural PCA"
    target = "qpos PCA target" if target_mode != "qpos" else "full qpos target"
    return f"{neural}, {target}"


def split_pretty_name(split_mode: str) -> str:
    """Make split names readable for plots."""
    return {
        "repeat_holdout": "repeat holdout",
        "gesture_holdout": "handle holdout",
        "session_holdout": "session holdout",
        "block_holdout": "block holdout",
    }.get(str(split_mode), str(split_mode))


def append_trial_metrics(
    out_dir: Path,
    *,
    experiment_id: str,
    row: dict,
    trials: list[Trial],
    preds: list[np.ndarray],
    sample_rate_hz: float,
    output_variant: str,
) -> None:
    """Append per-trial metrics for later mean/std and appendix plots."""
    rows = []
    for trial, pred in zip(trials, preds, strict=True):
        n = min(trial.qpos.shape[0], pred.shape[0])
        metrics = eval_metric_extras(trial.qpos[:n], pred[:n], sample_rate_hz=sample_rate_hz)
        rows.append(
            {
                "experiment_id": experiment_id,
                "output_variant": output_variant,
                "model_family": row.get("model_family"),
                "model_mode": row.get("model_mode"),
                "condition_object": row.get("condition_object"),
                "policy_label": policy_label(row),
                "pca_mode": row.get("pca_mode"),
                "target_mode": row.get("target_mode"),
                "pca_condition": pca_condition_label(row),
                "split_mode": row.get("split_mode"),
                "split_label": split_pretty_name(str(row.get("split_mode"))),
                "session": trial.session,
                "session_date": session_date_from_name(trial.session),
                "block": block_index_from_name(trial.session),
                "gesture": int(trial.gesture),
                "handle_id": None if trial.handle_id is None else int(trial.handle_id),
                "object_name": trial.object_name,
                "object_id": int(trial.object_id),
                "repeat": int(trial.repeat),
                "n_time": int(n),
                **metrics,
            }
        )
    if not rows:
        return
    path = out_dir / "trial_metrics.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def write_group_metric_summaries(out_dir: Path) -> pd.DataFrame:
    """Create mean/std metric summaries from trial_metrics.csv."""
    path = out_dir / "trial_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    trial_metrics = pd.read_csv(path)
    raw = trial_metrics[trial_metrics["output_variant"] == "raw"].copy()
    if raw.empty:
        return pd.DataFrame()
    metric_cols = [
        col
        for col in ["R2", "CC", "RMSE", "procrustes_corr", "jitter", "onset_delay_s", "offset_delay_s"]
        if col in raw.columns
    ]
    grouping_sets = {
        "session_date": ["experiment_id", "session_date"],
        "session_block": ["experiment_id", "session"],
        "object": ["experiment_id", "object_name"],
        "split_model_policy": ["split_mode", "model_family", "policy_label", "pca_condition"],
    }
    summary_frames = []
    for group_name, group_cols in grouping_sets.items():
        pieces = []
        grouped = raw.groupby(group_cols, dropna=False)
        for metric in metric_cols:
            stat = grouped[metric].agg(["mean", "std", "count"]).reset_index()
            stat["metric"] = metric
            stat["grouping"] = group_name
            stat = stat.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n_trials"})
            pieces.append(stat)
        if pieces:
            summary_frames.append(pd.concat(pieces, ignore_index=True))
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(out_dir / "metric_mean_std_by_group.csv", index=False)
    return summary


def plot_suite(out_dir: Path, leaderboard: pd.DataFrame, history: pd.DataFrame, trial_table: pd.DataFrame) -> None:
    """Create the standard VM plots saved in the result folder."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if leaderboard.empty:
        return

    leaderboard = leaderboard.copy()
    leaderboard["policy_label"] = [policy_label(r._asdict()) for r in leaderboard.itertuples(index=False)]
    leaderboard["pca_condition"] = [pca_condition_label(r._asdict()) for r in leaderboard.itertuples(index=False)]
    leaderboard["split_label"] = [split_pretty_name(x) for x in leaderboard["split_mode"]]

    top = leaderboard.sort_values("test_R2", ascending=False).head(24).iloc[::-1]
    labels = [
        f"{r.model_family}|{r.model_mode}|{r.split_mode}|{r.pca_mode}|{getattr(r, 'target_mode', 'qpos')}"
        for r in top.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(top))))
    ax.barh(labels, top["test_R2"], color="#2f80ed")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Test qpos R2")
    ax.set_title("Best decoder rows across split and PCA regimes")
    fig.tight_layout()
    fig.savefig(plot_dir / "leaderboard_test_r2.png", dpi=180)
    plt.close(fig)

    metrics = [
        "test_R2",
        "test_CC",
        "test_RMSE",
        "test_procrustes_corr",
        "test_jitter",
        "test_1euro_R2",
        "latency_ms_per_bin_offline",
        "test_onset_delay_s",
        "test_offset_delay_s",
    ]
    metrics = [m for m in metrics if m in leaderboard.columns]
    heat = leaderboard.sort_values("test_R2", ascending=False).head(30).set_index("experiment_id")[metrics]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(heat))))
    arr = heat.to_numpy(dtype=float)
    norm = np.empty_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if finite.any() and np.nanmax(col) > np.nanmin(col):
            norm[:, j] = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col))
        else:
            norm[:, j] = 0.5
    im = ax.imshow(norm, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels(heat.index, fontsize=7)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_title("Normalized metric heatmap, top rows")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(plot_dir / "decoder_metric_heatmap.png", dpi=180)
    plt.close(fig)

    pivot_index = ["model_family"]
    if "target_mode" in leaderboard.columns:
        pivot_index.append("target_mode")
    piv = leaderboard.pivot_table(index=pivot_index, columns=["split_mode", "pca_mode"], values="test_procrustes_corr", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([" / ".join(map(str, idx if isinstance(idx, tuple) else (idx,))) for idx in piv.index])
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{a}\n{b}" for a, b in piv.columns], rotation=0)
    ax.set_title("Best Procrustes trajectory-shape correlation")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            value = piv.to_numpy(dtype=float)[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(plot_dir / "procrustes_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    by_model = leaderboard.groupby("model_family")["train_ms"].median().sort_values()
    ax.bar(by_model.index, by_model.values / 1000.0, color="#f2994a")
    ax.set_ylabel("Median training time, seconds")
    ax.set_title("Training time by model family")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_time_by_model.png", dpi=180)
    plt.close(fig)

    if not history.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, group in history.groupby("experiment_id"):
            ax.plot(group["epoch"], group["val_loss"], label=str(key)[:42], linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.set_title("Validation loss trajectories for neural sequence models")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "loss_trajectories_overfitting_check.png", dpi=180)
        plt.close(fig)

    session_counts = trial_table.groupby("session").size().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(session_counts.index, session_counts.values, color="#00a6d6")
    ax.set_ylabel("Attempt trials")
    ax.set_title("Training data loaded by session")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "data_trials_by_session.png", dpi=180)
    plt.close(fig)

    # Main manuscript-style condition plot: y-axis is the metric, x-axis is the
    # split, and marker/color encode model/policy/PCA information.
    colors = {
        "ridge": "#4477aa",
        "ridge_qpos_pca": "#66c2a5",
        "kalman": "#cc6677",
        "refit_kalman": "#aa3377",
        "gru": "#228833",
        "lfads_lite": "#ee7733",
        "transformer": "#8856a7",
    }
    markers = {
        "direct no label": "o",
        "direct + object": "s",
        "hybrid object template": "^",
        "handle-known template": "D",
    }
    split_order = ["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"]
    split_order = [s for s in split_order if s in set(leaderboard["split_mode"])]
    x_positions = {split: i for i, split in enumerate(split_order)}
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in leaderboard.iterrows():
        x = x_positions.get(row["split_mode"], 0)
        x += (hash(str(row["model_family"]) + str(row["policy_label"]) + str(row["pca_condition"])) % 100 - 50) / 650.0
        ax.scatter(
            x,
            row["test_R2"],
            color=colors.get(str(row["model_family"]), "#555555"),
            marker=markers.get(str(row["policy_label"]), "o"),
            s=52 if str(row["target_mode"]) == "qpos" else 34,
            alpha=0.78 if str(row["pca_mode"]) == "none" else 0.48,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([x_positions[s] for s in split_order])
    ax.set_xticklabels([split_pretty_name(s) for s in split_order])
    ax.set_ylabel("Test R2")
    ax.set_title("Decoder comparison across split, model, object, and PCA conditions")
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in colors.items()
        if k in set(leaderboard["model_family"])
    ]
    policy_handles = [
        plt.Line2D([0], [0], marker=m, color="black", label=label, linestyle="", markersize=7)
        for label, m in markers.items()
        if label in set(leaderboard["policy_label"])
    ]
    legend1 = ax.legend(handles=model_handles, title="model", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=policy_handles, title="policy", loc="lower left", bbox_to_anchor=(1.01, 0.0))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "main_r2_condition_scatter.png", dpi=180)
    plt.close(fig)

    # Mean/std summaries from per-trial metrics.  These figures become useful
    # once the full VM run has completed and trial_metrics.csv exists.
    summary = write_group_metric_summaries(out_dir)
    trial_metric_path = out_dir / "trial_metrics.csv"
    if trial_metric_path.exists():
        trial_metrics = pd.read_csv(trial_metric_path)
        raw = trial_metrics[trial_metrics["output_variant"] == "raw"].copy()
        if not raw.empty:
            best_ids = leaderboard.sort_values("test_R2", ascending=False).head(12)["experiment_id"].tolist()
            top_raw = raw[raw["experiment_id"].isin(best_ids)].copy()
            if not top_raw.empty:
                order = best_ids[::-1]
                fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(order))))
                y_positions = np.arange(len(order))
                labels = []
                means = []
                stds = []
                for exp_id in order:
                    vals = top_raw.loc[top_raw["experiment_id"] == exp_id, "R2"].to_numpy(dtype=float)
                    means.append(float(np.nanmean(vals)))
                    stds.append(float(np.nanstd(vals)))
                    row = leaderboard.loc[leaderboard["experiment_id"] == exp_id].iloc[0]
                    labels.append(f"{row['model_family']} | {row['policy_label']} | {row['split_label']} | {row['pca_condition']}")
                ax.errorbar(means, y_positions, xerr=stds, fmt="o", color="#2a9d8f", ecolor="#777777", capsize=3)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel("Per-trial R2, mean +/- std")
                ax.set_title("Top decoder rows with trial-level variability")
                ax.grid(alpha=0.25, axis="x")
                fig.tight_layout()
                fig.savefig(plot_dir / "top_models_r2_mean_std.png", dpi=180)
                plt.close(fig)

            object_summary = (
                raw.groupby(["object_name", "model_family"], dropna=False)["R2"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            if not object_summary.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                objects = sorted(object_summary["object_name"].dropna().unique())
                families = [f for f in ["ridge", "kalman", "refit_kalman", "gru", "lfads_lite", "transformer"] if f in set(object_summary["model_family"])]
                width = 0.8 / max(1, len(families))
                xs = np.arange(len(objects))
                for i, family in enumerate(families):
                    vals = []
                    errs = []
                    for obj in objects:
                        subset = object_summary[(object_summary["object_name"] == obj) & (object_summary["model_family"] == family)]
                        vals.append(float(subset["mean"].iloc[0]) if not subset.empty else np.nan)
                        errs.append(float(subset["std"].iloc[0]) if not subset.empty else np.nan)
                    ax.bar(xs + (i - len(families) / 2) * width + width / 2, vals, width=width, yerr=errs, label=family, alpha=0.82)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xticks(xs)
                ax.set_xticklabels(objects, rotation=25, ha="right")
                ax.set_ylabel("Per-trial R2, mean +/- std")
                ax.set_title("Object-wise decoding performance")
                ax.legend(ncol=3, fontsize=8)
                fig.tight_layout()
                fig.savefig(plot_dir / "object_r2_mean_std_by_model.png", dpi=180)
                plt.close(fig)

            appendix_dir = plot_dir / "appendix_per_session"
            appendix_dir.mkdir(parents=True, exist_ok=True)
            for date, group in raw.groupby("session_date"):
                show_ids = (
                    group.groupby("experiment_id")["R2"].mean().sort_values(ascending=False).head(12).index.tolist()
                )
                session_group = group[group["experiment_id"].isin(show_ids)].copy()
                if session_group.empty:
                    continue
                stats = session_group.groupby("experiment_id")["R2"].agg(["mean", "std"]).loc[show_ids].iloc[::-1]
                fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(stats))))
                ax.errorbar(stats["mean"], np.arange(len(stats)), xerr=stats["std"], fmt="o", color="#4477aa", capsize=3)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(np.arange(len(stats)))
                ax.set_yticklabels(stats.index, fontsize=7)
                ax.set_xlabel("Per-trial R2, mean +/- std")
                ax.set_title(f"Appendix: performance for session {date}")
                ax.grid(alpha=0.25, axis="x")
                fig.tight_layout()
                fig.savefig(appendix_dir / f"session_{date}_r2_mean_std.png", dpi=180)
                plt.close(fig)


def write_report(out_dir: Path, leaderboard: pd.DataFrame, trial_table: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write the raw training README with commands, sessions, and leaderboards."""
    best_cols = [
        "experiment_id",
        "model_family",
        "model_mode",
        "split_mode",
        "pca_mode",
        "target_mode",
        "condition_object",
        "template_key_name",
        "target_pca_components",
        "target_pca_explained_variance",
        "qpos_pca_components",
        "qpos_pca_explained_variance",
        "test_R2",
        "test_CC",
        "test_RMSE",
        "test_procrustes_corr",
        "test_jitter",
        "test_1euro_R2",
        "test_1euro_CC",
        "test_1euro_RMSE",
        "latency_ms_per_bin_offline",
        "test_onset_delay_s",
        "test_offset_delay_s",
        "train_ms",
        "mujoco_compare_dir",
        "mujoco_compare_1euro_dir",
    ]
    best_cols = [c for c in best_cols if c in leaderboard.columns]
    lines = [
        "# Same-Threshold Split/PCA Patient Decoder Suite Results",
        "",
        "This run trains patient-only decoders on the session folders passed through `--data-root`. No FALCON data are used. For the same-threshold rerun, the intended folders are the valid VM attempt blocks directly under `/mnt/vmshare`; the nested folder named `thresholds defined per block` is intentionally ignored.",
        "",
        "## What was tested",
        "",
        "- `repeat_holdout`: the existing split style. Repetitions 0-5 are used for training, 6-7 for validation, and 8-9 for testing.",
        "- `gesture_holdout`: all repetitions of selected handle IDs are withheld from training. This is the stricter unseen-handle/unseen-gesture test.",
        f"- `session_holdout`: one full recording date is withheld for test. If no date is specified, the latest discovered date is held out; this run used `{args.session_holdout_test_date or 'latest discovered date'}`.",
        f"- `block_holdout`: one block index is withheld across sessions. This run used block `{args.block_holdout_test_block}` as the test block and the remaining blocks for training/validation.",
        "- `pca=none`: models receive the causal neural feature space directly.",
        f"- `pca=pca{args.pca_components}`: PCA is fit on training neural bins only, then applied to validation/test. This avoids test leakage.",
        "- `target=qpos`: models predict the full 32-D qpos/residual target directly.",
        f"- `target=qpos_pca`: when requested, Ridge/GRU/LFADS-lite/Transformer predict {args.qpos_pca_components} kinematic target PCs and reconstruct the full 32-D qpos before scoring/export.",
        "- `pca_diagnostics/`: when PCA is requested, the VM also saves a scree/cumulative-variance plot fitted on training neural bins only.",
        "- optional `ridge_qpos_pca`: if requested, Ridge predicts a smaller number of qpos principal-component scores and reconstructs the full 32-D qpos target before scoring.",
        "- optional `handle_known` Ridge: if requested, repeat-holdout Ridge uses the exact handle/gesture template prior. This is a ceiling-style diagnostic, not a label-free decoder.",
        "- Models include Ridge, Kalman, ReFIT-style Kalman, GRU, LFADS-lite, and causal Transformer variants.",
        "",
        "## Correct object mapping used",
        "",
        "The same-threshold VM pickle files do not store an explicit `Object.Id` column. The object labels are therefore reconstructed from the protocol mapping supplied for this rerun:",
        "",
        "1. The group order in `all_attempt.pkl` / `trials-attempt-resampled.pkl` is mapped to the true handle IDs for the session date and block.",
        "2. Each handle ID is mapped to a MuJoCo geom/body ID.",
        "3. The geom/body ID is mapped to the six canonical object names: `small_ball`, `big_ball`, `small_v_cylinder`, `big_v_cylinder`, `small_h_cylinder`, `big_h_cylinder`.",
        "4. The resulting object name is converted into the one-hot object label used by object-conditioned and hybrid models.",
        "",
        "The handle-to-geom mapping used by the loader is:",
        "",
        markdown_table(pd.DataFrame({"handle_id": list(PROTOCOL_HANDLE_TO_GEOM_ID.keys()), "geom_id": list(PROTOCOL_HANDLE_TO_GEOM_ID.values()), "object_name": [VM_BODY_ID_TO_OBJECT[g] for g in PROTOCOL_HANDLE_TO_GEOM_ID.values()]})),
        "",
        "For folders containing six gesture groups, the block number in the folder name selects the corresponding six-handle slice from the 18-handle session order. For folders containing all 18 gesture groups, the full date-level order is used.",
        "",
        "Object-label counts in the loaded data:",
        "",
        markdown_table(trial_table.groupby(["object_name", "object_id", "object_label_source"]).size().reset_index(name="trials")),
        "",
        "## Model inputs and outputs",
        "",
        "- Input neural data are attempted-movement bins at 100 Hz. Each bin contains the 256-channel neural feature vector after causal smoothing.",
        "- Direct models receive neural features plus elapsed phase. Object-conditioned direct rows also receive a one-hot object label from the data loader.",
        "- Hybrid models receive neural features plus elapsed phase plus a training-set template/prior for the object/gesture. The model predicts a residual correction around that prior, so hybrid rows are task-assisted rather than fully label-free.",
        "- The decoded output is always 32-D robot `qpos` in radians. MuJoCo replay folders also include mapped `ctrl` arrays for visualization.",
        "- Full-profile neural rows use hidden size 96, batch size 32, AdamW (`lr=1e-3`, `weight_decay=1e-4`), at most 70 epochs, and early stopping with patience 12. LFADS-lite uses a 32-D factor bottleneck; the causal Transformer uses two layers and four attention heads.",
        "",
        "## Important interpretation",
        "",
        "The strongest previous models were task-assisted. In this suite, `hybrid` rows use an object-known template prior plus a neural residual. `direct` rows predict qpos directly from neural history and optional object conditioning. `gesture_holdout` is expected to be much harder, especially for hybrid models, because exact movement templates from the held-out gesture are not available.",
        "",
        "This VM package defaults to a 100 Hz time base, i.e. one neural/kinematic bin is interpreted as 10 ms. This affects timing metrics such as onset delay, offset delay, and latency normalization; it does not upsample or downsample the arrays by itself.",
        "",
        "## Metric definitions",
        "",
        "- `test_R2`: variance explained for 32-D qpos. Higher is better; 1.0 is perfect, 0 means no better than predicting the mean.",
        "- `test_CC`: average temporal correlation between actual and decoded joint traces. Higher means the decoder follows the shape of the movement better.",
        "- `test_RMSE`: root mean squared qpos error in radians. Lower is better.",
        "- `test_procrustes_corr`: centered, scale-normalized trajectory-shape similarity. Higher is better and is useful when the absolute offset/scale is imperfect but the movement shape is similar. This implementation does not perform a full rotation-aligned 3-D Procrustes fit.",
        "- `test_jitter`: roughness of the decoded trajectory from bin to bin. Lower means smoother output.",
        "- `test_onset_delay_s` and `test_offset_delay_s`: movement start/end timing error in seconds.",
        "- `test_best_lag_bins` / `test_best_lag_R2`: diagnostic timing-shift check. These are not used to train the model; they show whether the decoder is systematically early or late.",
        "- `latency_ms_per_bin_offline`: measured inference time normalized per 100 Hz bin on the machine running the suite.",
        "- `test_1euro_*`: the same metrics after applying the causal 1Euro output filter to the decoded qpos. These are for online-smoothing/viz comparison; the raw decoder score is still kept.",
        "- `qpos_pca_explained_variance`: for optional qpos-PCA Ridge only, the fraction of target qpos variance represented by the retained kinematic components.",
        "- `target_pca_explained_variance`: for labelled `target=qpos_pca` rows, the fraction of training target variance represented by the retained kinematic components.",
        "",
        "## Sessions loaded",
        "",
        markdown_table(trial_table.groupby(["session"]).agg(trials=("repeat", "size"), gestures=("gesture", "nunique"), handles=("handle_id", "nunique")).reset_index()),
        "",
        "## Best rows",
        "",
        markdown_table(leaderboard.sort_values("test_R2", ascending=False)[best_cols], max_rows=30),
        "",
        "## Plots",
        "",
        "- `plots/leaderboard_test_r2.png`: top rows by qpos R2.",
        "- `plots/decoder_metric_heatmap.png`: normalized heatmap for R2, CC, RMSE, Procrustes, jitter, onset and offset delay.",
        "- `plots/procrustes_heatmap.png`: trajectory-shape similarity by model family, split, and PCA mode.",
        "- `plots/training_time_by_model.png`: median training time by family.",
        "- `plots/loss_trajectories_overfitting_check.png`: validation loss curves for GRU/LFADS/Transformer rows.",
        "- `plots/data_trials_by_session.png`: number of attempt trials loaded per session.",
        "- `plots/main_r2_condition_scatter.png`: compact manuscript-style overview. The y-axis is test R2; the x-axis is split type; color encodes model family; marker encodes direct/object/hybrid policy.",
        "- `plots/top_models_r2_mean_std.png`: top decoder rows with mean and standard deviation across test trials.",
        "- `plots/object_r2_mean_std_by_model.png`: object-wise R2 with mean and standard deviation.",
        "- `plots/appendix_per_session/session_*_r2_mean_std.png`: one figure per session date for appendix-style reporting.",
        "- `pca_diagnostics/*_neural_pca_explained_variance.png`: full neural PCA explained-variance curves for each split.",
        "- `metric_mean_std_by_group.csv`: numerical mean/std table grouped by session date, session block, object, and split/model/policy.",
        "- `trial_metrics.csv`: one row per exported test trial and model condition, useful for custom manuscript figures.",
        "",
        "## Manuscript plotting suggestion",
        "",
        "- Main results figure: use `plots/main_r2_condition_scatter.png` or a simplified version of it, because it compresses model family, split difficulty, label condition, and PCA condition into one readable comparison.",
        "- Main robustness figure: use `plots/top_models_r2_mean_std.png`, because it shows both central tendency and variability.",
        "- Object question from review: use `plots/object_r2_mean_std_by_model.png`, then keep the underlying `metric_mean_std_by_group.csv` table for exact numbers.",
        "- Appendix: use the per-session figures in `plots/appendix_per_session/` to show individual session/date behavior.",
        "",
        "## MuJoCo visualization",
        "",
        "Each trained row writes two MuJoCo replay folders:",
        "",
        "- `mujoco_compare_dir`: raw decoded output.",
        "- `mujoco_compare_1euro_dir`: the same decoded output after a causal 1Euro filter (`min_cutoff=1.0`, `beta=0.02`, `d_cutoff=1.0`, `dt=0.01`).",
        "",
        "After downloading the artifact folder to the Mac, run for example:",
        "",
        "```bash",
        "cd /path/to/local/thesis/review/package",
        "mjpython scripts/view_mujoco_hand.py --compare --compare-dir /path/to/mujoco_compare_dir --repeat 8 --camera-view arm --factor 120 --slow-loop-factor 1200 --loop",
        "```",
        "",
        "For gesture-holdout rows, the repeat number in the file names may not be 8 for every exported trial; the viewer still reads the folder catalog.",
        "",
        "## BRAND/online-readiness notes",
        "",
        "- Ridge and Kalman are fastest and easiest to deploy, but may be less accurate.",
        "- GRU and LFADS-lite are causal when `bidirectional=false` and can be deployed if latency is acceptable.",
        "- Causal Transformer rows use a causal attention mask. They are still heavier and need latency testing.",
        "- Bidirectional LFADS, if enabled, is an offline ceiling only because it uses future context.",
        "- Object-conditioned rows require the task software or an object classifier to supply the object label online. For six-object VM models, the streaming node accepts compact IDs `0..5` or body IDs `55..60` and maps them internally.",
        "",
    ]
    (out_dir / "TRAINING_RESULTS_README.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    experiment_id: str,
    model_family: str,
    model_mode: str,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    projector: NeuralProjector,
    n_objects: int,
    template_bank: TemplateBank | None,
    condition_object: bool,
    target_projector: TargetProjector,
    template_key_name: str,
    qpos_pca_components: int,
    split_mode: str,
    pca_mode: str,
    device: str,
    profile: str,
    seed: int,
    bidirectional_lfads: bool,
    gru_layers: int,
    sample_rate_hz: float,
    out_dir: Path,
) -> tuple[dict, list[np.ndarray], pd.DataFrame]:
    """Run one split/PCA/model configuration and export metrics plus MuJoCo arrays."""
    # One "experiment" is exactly one row in the leaderboard, for example:
    # repeat_holdout + pca64 + gru + hybrid + object-conditioned.
    model_out = out_dir / "models" / experiment_id
    model_out.mkdir(parents=True, exist_ok=True)
    if model_family == "ridge":
        row, preds = train_eval_ridge_family(
            train,
            val,
            test,
            projector,
            n_objects,
            model_mode=model_mode,
            template_bank=template_bank,
            condition_object=condition_object,
            target_projector=target_projector,
            sample_rate_hz=sample_rate_hz,
            template_key_name=template_key_name,
        )
        history = pd.DataFrame()
    elif model_family == "ridge_qpos_pca":
        row, preds = train_eval_qpos_pca_ridge_family(
            train,
            val,
            test,
            projector,
            n_objects,
            condition_object=condition_object,
            sample_rate_hz=sample_rate_hz,
            qpos_pca_components=qpos_pca_components,
            seed=seed,
        )
        history = pd.DataFrame()
    elif model_family in {"gru", "lfads_lite", "transformer"}:
        row, preds, history = train_sequence_decoder(
            train,
            val,
            test,
            projector,
            n_objects,
            model_family=model_family,
            model_mode=model_mode,
            template_bank=template_bank,
            condition_object=condition_object,
            target_projector=target_projector,
            template_key_name=template_key_name,
            device_name=device,
            profile=profile,
            seed=seed,
            bidirectional_lfads=bidirectional_lfads,
            gru_layers=gru_layers,
            sample_rate_hz=sample_rate_hz,
            out_dir=model_out,
        )
    elif model_family in {"kalman", "refit_kalman"}:
        row, preds, _train_ms = train_eval_kalman(
            train,
            val,
            test,
            projector,
            refit=model_family == "refit_kalman",
            sample_rate_hz=sample_rate_hz,
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
        "target_mode": target_projector.mode,
        "target_pca_components": target_projector.output_dim if target_projector.mode != "qpos" else np.nan,
        "target_pca_explained_variance": target_projector.explained_variance if target_projector.mode != "qpos" else np.nan,
        "template_key_name": template_key_name,
        "n_train_trials": len(train),
        "n_val_trials": len(val),
        "n_test_trials": len(test),
    }
    filtered_preds = one_euro_filter_trials(preds, sample_rate_hz=sample_rate_hz)
    filtered_metrics = aggregate_trial_metrics(test, filtered_preds, sample_rate_hz=sample_rate_hz)
    row.update({f"test_1euro_{k}": v for k, v in filtered_metrics.items()})
    row.update(
        {
            "one_euro_min_cutoff": ONE_EURO_MIN_CUTOFF,
            "one_euro_beta": ONE_EURO_BETA,
            "one_euro_d_cutoff": ONE_EURO_D_CUTOFF,
        }
    )
    append_trial_metrics(
        out_dir,
        experiment_id=experiment_id,
        row=row,
        trials=test,
        preds=preds,
        sample_rate_hz=sample_rate_hz,
        output_variant="raw",
    )
    append_trial_metrics(
        out_dir,
        experiment_id=experiment_id,
        row=row,
        trials=test,
        preds=filtered_preds,
        sample_rate_hz=sample_rate_hz,
        output_variant="1euro",
    )
    compare_dir = out_dir / "mujoco_compare" / experiment_id
    # The visualization export always compares test-set actual movement against
    # model-decoded movement.  The files are saved as ctrl arrays because the
    # MuJoCo viewer consumes ctrl order, even though the model output was qpos.
    export_mujoco_compare(test, preds, compare_dir, experiment_id, max_trials=24)
    row["mujoco_compare_dir"] = str(compare_dir)
    filtered_compare_dir = out_dir / "mujoco_compare_1euro" / experiment_id
    export_mujoco_compare(test, filtered_preds, filtered_compare_dir, f"{experiment_id}__1euro", max_trials=24)
    row["mujoco_compare_1euro_dir"] = str(filtered_compare_dir)
    if not history.empty:
        history["experiment_id"] = experiment_id
        history["split_mode"] = split_mode
        history["pca_mode"] = pca_mode
    return clean_dict(row), preds, history


def main() -> int:
    """Load sessions, loop over split/PCA/model configurations, and save outputs."""
    parser = argparse.ArgumentParser(description="Run split x PCA x model-family patient decoder suite.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BUNDLE_ROOT / "artifacts" / "split_pca_model_suite",
        help="Folder where metrics, model files, plots, training curves, logs, and MuJoCo comparison exports are saved.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=None,
        help="Patient session/block folder to load. Pass one --data-root per session; the wrapper normally fills this automatically.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="Folder containing named template trajectories used by hybrid/template-assisted decoders.",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "quick", "full"],
        default="quick",
        help="Training budget preset. smoke checks the pipeline quickly; full is the intended thesis run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for neural models. auto chooses cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"],
        choices=["repeat_holdout", "gesture_holdout", "session_holdout", "block_holdout"],
        help=(
            "Train/test protocols. repeat_holdout tests later repetitions; "
            "gesture_holdout holds out entire handle IDs; session_holdout holds out one recording date; "
            "block_holdout trains on two blocks and tests on one block."
        ),
    )
    parser.add_argument(
        "--pca-modes",
        nargs="+",
        default=["none", "pca"],
        choices=["none", "pca"],
        help="Neural feature modes. none keeps original causal features; pca uses training-only PCA before decoding.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="Number of PCA components when pca mode is enabled; pca64 is the default comparison setting.",
    )
    parser.add_argument(
        "--target-modes",
        nargs="+",
        default=["qpos"],
        choices=["qpos", "qpos_pca"],
        help=(
            "Target representation modes. qpos predicts all 32 qpos values directly. "
            "qpos_pca predicts kinematic PC scores and reconstructs 32-D qpos before metrics/export."
        ),
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Sampling rate used for timing metrics and lag conversion. 100 Hz means one bin is 10 ms.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260709,
        help="Random seed for deterministic gesture holdout, PCA subsampling, and neural-model initialization.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Optional debugging limit on the number of model configurations to run; leave unset for complete training.",
    )
    parser.add_argument(
        "--bidirectional-lfads",
        action="store_true",
        help="Use a bidirectional LFADS-lite encoder. This can improve offline accuracy but is not strictly causal/online.",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=1,
        help="Number of recurrent GRU layers for GRU rows. Default 1 preserves the completed VM run; use 2 for the BRAND candidate experiment.",
    )
    parser.add_argument(
        "--include-handle-ceiling",
        action="store_true",
        help=(
            "Also run a repeat-holdout Ridge row that uses the exact handle_id template prior. "
            "This reproduces the earlier handle-known/template-ceiling idea and is skipped for gesture_holdout."
        ),
    )
    parser.add_argument(
        "--include-qpos-pca-ridge",
        action="store_true",
        help=(
            "Also run Ridge rows that predict low-dimensional qpos principal components, "
            "then reconstruct the full 32-D qpos output."
        ),
    )
    parser.add_argument(
        "--qpos-pca-components",
        type=int,
        default=12,
        help="Number of kinematic/qpos PCA components for --target-modes qpos_pca or --include-qpos-pca-ridge.",
    )
    parser.add_argument(
        "--session-holdout-test-date",
        default="",
        help="YYYYMMDD date held out by session_holdout. Empty means the latest discovered date.",
    )
    parser.add_argument(
        "--block-holdout-test-block",
        type=int,
        default=3,
        help="Block index held out by block_holdout. Default 3.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # The wrapper normally passes one --data-root per session/block.  If this
    # script is run directly with no data roots, it falls back to the three old
    # bundled patient sessions.
    data_roots = args.data_root if args.data_root is not None else [p for p in OLD_PATIENT_ROOTS if p.exists()]
    # combine_sessions() is the bridge from mixed disk formats to the common
    # Trial list.  From this point onward the code no longer cares whether a
    # trial came from 0116, 0123/0130, or a VM block.
    dataset, trial_table = combine_sessions(list(data_roots), args.template_dir, args.out_dir)
    trial_table.to_csv(args.out_dir / "combined_trial_index.csv", index=False)
    # n_objects controls the width of object_onehot().  If object IDs are
    # 0..5, for example, the object label vector has six columns.
    n_objects = int(max(t.object_id for t in dataset.trials) + 1)
    rows = []
    histories = []
    experiment_count = 0

    # This is the model grid.  The same model family can appear in several
    # "policy modes":
    #   direct/nolabel: neural + phase -> qpos
    #   direct/obj: neural + phase + object one-hot -> qpos
    #   hybrid/obj: neural + phase + template prior + object one-hot -> residual
    # Kalman rows are direct and label-free in this suite.
    model_specs = [
        {"model_family": "ridge", "model_mode": "direct", "condition_object": False},
        {"model_family": "ridge", "model_mode": "direct", "condition_object": True},
        {"model_family": "ridge", "model_mode": "hybrid", "condition_object": True, "template_key_name": "object_id"},
        {"model_family": "kalman", "model_mode": "direct", "condition_object": False},
        {"model_family": "refit_kalman", "model_mode": "direct", "condition_object": False},
        {"model_family": "gru", "model_mode": "direct", "condition_object": False},
        {"model_family": "gru", "model_mode": "direct", "condition_object": True},
        {"model_family": "gru", "model_mode": "hybrid", "condition_object": True, "template_key_name": "object_id"},
        {"model_family": "lfads_lite", "model_mode": "direct", "condition_object": False},
        {"model_family": "lfads_lite", "model_mode": "hybrid", "condition_object": True, "template_key_name": "object_id"},
        {"model_family": "transformer", "model_mode": "direct", "condition_object": False},
    ]
    if args.profile == "smoke":
        model_specs = model_specs[:4]
    if args.include_handle_ceiling:
        model_specs.append(
            {
                "model_family": "ridge",
                "model_mode": "hybrid",
                "condition_object": True,
                "template_key_name": "handle_id",
                "name_suffix": "handle_known",
            }
        )
    if args.include_qpos_pca_ridge:
        model_specs.extend(
            [
                {
                    "model_family": "ridge_qpos_pca",
                    "model_mode": "direct_qpos_pca",
                    "condition_object": False,
                    "name_suffix": f"qpospc{args.qpos_pca_components}",
                },
                {
                    "model_family": "ridge_qpos_pca",
                    "model_mode": "direct_qpos_pca",
                    "condition_object": True,
                    "name_suffix": f"qpospc{args.qpos_pca_components}",
                },
            ]
        )

    for split_mode in args.splits:
        # Split choice decides which trials are allowed during training:
        # repeat_holdout = train early repeats, test later repeats.
        # gesture_holdout = train on some gesture IDs, test entirely unseen IDs.
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
        # Hybrid models may only build templates from training trials.  That is
        # important: validation/test qpos are not averaged into the prior.
        object_template_bank = build_template_bank(train, "object_id")
        handle_template_bank = build_template_bank(train, "handle_id") if args.include_handle_ceiling else None
        for pca_mode in args.pca_modes:
            # PCA mode decides the neural input size:
            #   none -> smoothed 256-D neural vector
            #   pca -> pca_components-D vector, fitted on training bins only
            projector = fit_neural_projector(
                train,
                mode="none" if pca_mode == "none" else "pca",
                components=args.pca_components,
                seed=args.seed,
            )
            for spec in model_specs:
                template_key_name = str(spec.get("template_key_name", "object_id"))
                if template_key_name == "handle_id" and split_mode != "repeat_holdout":
                    print("Skipping handle-known Ridge ceiling for gesture_holdout because the test handles are unseen.", flush=True)
                    continue
                template_bank = handle_template_bank if template_key_name == "handle_id" else object_template_bank
                for target_mode in args.target_modes:
                    if target_mode != "qpos" and spec["model_family"] in {"kalman", "refit_kalman"}:
                        print("Skipping qpos-PCA target for Kalman/ReFIT because those baselines use a full qpos+qvel state.", flush=True)
                        continue
                    if target_mode != "qpos" and spec["model_family"] == "ridge_qpos_pca":
                        # Backward-compatible old diagnostic; avoid applying
                        # the new target-PCA wrapper on top of its own PCA.
                        continue
                    target_projector = fit_target_projector(
                        train,
                        target_mode=target_mode,
                        model_mode=spec["model_mode"],
                        template_bank=template_bank if spec["model_mode"] == "hybrid" else None,
                        template_key_name=template_key_name,
                        components=args.qpos_pca_components,
                        seed=args.seed + experiment_count,
                    )
                    name_suffix = spec.get("name_suffix")
                    experiment_id = (
                        f"{split_mode}__{projector.mode}__{spec['model_family']}__"
                        f"{spec['model_mode']}__{'obj' if spec['condition_object'] else 'nolabel'}"
                    )
                    if target_projector.mode != "qpos":
                        experiment_id = f"{experiment_id}__target_{target_projector.mode}"
                    if name_suffix:
                        experiment_id = f"{experiment_id}__{name_suffix}"
                    experiment_count += 1
                    if args.max_experiments is not None and experiment_count > args.max_experiments:
                        break
                    print(f"\n=== {experiment_id} ===", flush=True)
                    row, _preds, history = run_experiment(
                        experiment_id=experiment_id,
                        model_family=spec["model_family"],
                        model_mode=spec["model_mode"],
                        train=train,
                        val=val,
                        test=test,
                        projector=projector,
                        n_objects=n_objects,
                        template_bank=template_bank if spec["model_mode"] == "hybrid" else None,
                        condition_object=bool(spec["condition_object"]),
                        target_projector=target_projector,
                        template_key_name=template_key_name,
                        qpos_pca_components=args.qpos_pca_components,
                        split_mode=split_mode,
                        pca_mode=projector.mode,
                        device=device,
                        profile=args.profile,
                        seed=args.seed + experiment_count,
                        bidirectional_lfads=args.bidirectional_lfads,
                        gru_layers=args.gru_layers,
                        sample_rate_hz=args.sample_rate_hz,
                        out_dir=args.out_dir,
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

    leaderboard = pd.DataFrame(rows).sort_values(["test_R2", "test_CC"], ascending=False)
    leaderboard.to_csv(args.out_dir / "leaderboard.csv", index=False)
    per_dim_rows = []
    for row in rows:
        # Per-dimension arrays are better computed from saved compare files in
        # the final analysis; this placeholder records where they can be found.
        per_dim_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "mujoco_compare_dir": row["mujoco_compare_dir"],
                "mujoco_compare_1euro_dir": row.get("mujoco_compare_1euro_dir", ""),
            }
        )
    pd.DataFrame(per_dim_rows).to_csv(args.out_dir / "per_dimension_sources.csv", index=False)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    if not history_df.empty:
        history_df.to_csv(args.out_dir / "training_history.csv", index=False)
    manifest = {
        "data_roots": [str(p) for p in data_roots],
        "splits": args.splits,
        "pca_modes": args.pca_modes,
        "pca_components": args.pca_components,
        "target_modes": args.target_modes,
        "profile": args.profile,
        "device": device,
        "sample_rate_hz": args.sample_rate_hz,
        "bidirectional_lfads": args.bidirectional_lfads,
        "gru_layers": args.gru_layers,
        "include_handle_ceiling": args.include_handle_ceiling,
        "include_qpos_pca_ridge": args.include_qpos_pca_ridge,
        "qpos_pca_components": args.qpos_pca_components,
        "session_holdout_test_date": args.session_holdout_test_date,
        "block_holdout_test_block": args.block_holdout_test_block,
        "uses_falcon": False,
        "notes": "This package retrains models from scratch on the loaded patient sessions so scalers, PCA bases, templates, and model weights are consistent across old and VM sessions.",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_suite(args.out_dir, leaderboard, history_df, trial_table)
    write_report(args.out_dir, leaderboard, trial_table, args)
    print(f"Wrote suite outputs to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
