"""Shared causal preprocessing, splits, metrics, and architectures for the final run."""
from __future__ import annotations
import math
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from decoder_core.data import Trial, causal_ema, split_trials
from decoder_core.metrics import summarize, rmse
from decoder_core.models import _fit_ridge_readout
N_QPOS=32
DT=.01
ONE_EURO_MIN_CUTOFF=1.0
ONE_EURO_BETA=.02
ONE_EURO_D_CUTOFF=1.0

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
    # movement conditions that were not represented in the training set.
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
    """Fixed elapsed-time clock, available at onset without final trial length."""
    return (np.arange(trial.n_time, dtype=np.float32) / 600.0)[:, None]


def eval_metric_extras(y_true: np.ndarray, y_pred: np.ndarray, *, sample_rate_hz: float) -> dict[str, float]:
    """Per-trial metrics. Call temporal_metrics_by_trial for pooled sequences."""
    base = summarize(y_true, y_pred)
    onset, offset = onset_offset_delay(y_true, y_pred, sample_rate_hz=sample_rate_hz)
    a = np.array(y_true, dtype=np.float64, copy=True); b = np.array(y_pred, dtype=np.float64, copy=True)
    a -= a.mean(axis=0); b -= b.mean(axis=0)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    aligned = float(np.linalg.svd(a.T @ b, compute_uv=False).sum() / denom) if denom > 1e-12 else float('nan')
    energy_true = movement_energy(y_true); energy_pred = movement_energy(y_pred)
    def sustained_bounds(e):
        threshold = .2 * np.percentile(e, 99)
        if threshold <= 1e-12: return (float('nan'), float('nan'))
        active = e >= threshold
        runs = np.where(np.convolve(active.astype(int), np.ones(5, dtype=int), mode='valid') == 5)[0]
        return (float(runs[0]), float(runs[-1] + 4)) if len(runs) else (float('nan'), float('nan'))
    ts, te = sustained_bounds(energy_true); ps, pe = sustained_bounds(energy_pred)
    best_lag, best_r2 = lag_sensitivity(y_true, y_pred)
    return {**base, 'procrustes_corr': procrustes_corr(y_true, y_pred),
            'procrustes_disparity': procrustes_disparity(y_true, y_pred),
            'centered_normalized_corr': procrustes_corr(y_true, y_pred),
            'orthogonal_procrustes_corr': aligned,
            'centered_trajectory_similarity': procrustes_corr(y_true, y_pred),
            # Standard similarity Procrustes: optimal rotation/reflection and scale.
            # With centered unit-norm inputs, disparity = 1 - aligned_corr**2.
            'standard_procrustes_disparity': max(0.0, 1.0 - aligned**2) if math.isfinite(aligned) else float('nan'),
            'onset_delay_s': onset, 'offset_delay_s': offset,
            'sustained_onset_delay_s': (ps-ts)/sample_rate_hz,
            'sustained_offset_delay_s': (pe-te)/sample_rate_hz,
            'initial_100ms_RMSE': rmse(y_true[:max(1,int(.1*sample_rate_hz))], y_pred[:max(1,int(.1*sample_rate_hz))]),
            'initial_prediction_step_norm': float(energy_pred[1]) if len(energy_pred)>1 else 0.,
            'best_lag_bins': best_lag, 'best_lag_R2': best_r2}


def temporal_metrics_by_trial(true_trials, pred_trials, sample_rate_hz):
    """Pooled accuracy plus equal-trial temporal summaries; no boundary differences."""
    true = np.concatenate(true_trials); pred = np.concatenate(pred_trials)
    result = eval_metric_extras(true, pred, sample_rate_hz=sample_rate_hz)
    details = [eval_metric_extras(a,b,sample_rate_hz=sample_rate_hz)
               for a,b in zip(true_trials,pred_trials,strict=True)]
    for key in ['jitter','drift','onset_delay_s','offset_delay_s','sustained_onset_delay_s',
                'sustained_offset_delay_s','initial_100ms_RMSE','initial_prediction_step_norm',
                'best_lag_bins','best_lag_R2']:
        values = np.asarray([r[key] for r in details]); values = values[np.isfinite(values)]
        result[key] = float(values.mean()) if len(values) else float('nan')
        result[key+'_std'] = float(values.std(ddof=1)) if len(values)>1 else 0.
        result[key+'_valid_trials'] = int(len(values))
    return result


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


def kalman_gains(params, n_time):
    """Reuse data-independent gains across trials; solve in state space, not channels."""
    cached = params.get('_gain_cache', [])
    A,H = params['A'].astype(float),params['H'].astype(float)
    eye = np.eye(A.shape[0]); Q = np.diag(params['Q_diag']); rinv = 1./params['R_diag']
    ht_rinv = H.T * rinv[None,:]
    information = ht_rinv @ H
    p = params.get('_last_cov',eye.copy())
    for _ in range(len(cached),n_time):
        prior = A @ p @ A.T + Q
        p = np.linalg.solve(np.linalg.solve(prior,eye) + information,eye)
        p = (p+p.T)/2
        cached.append((p @ ht_rinv).astype(np.float32))
    params['_gain_cache']=cached; params['_last_cov']=p
    return cached


def kalman_filter_trial(params: dict, obs: np.ndarray) -> np.ndarray:
    """Causal Kalman pass; gains/covariances depend on time, never on test targets."""
    z = params['z_mean'].astype(np.float32).copy()
    out = np.empty((len(obs),len(z)),dtype=np.float32)
    for t,K in enumerate(kalman_gains(params,len(obs))[:len(obs)]):
        zpred = params['A'] @ z + params['A_b']
        z = zpred + K @ (obs[t] - (params['H'] @ zpred + params['H_b']))
        out[t]=z
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


def qpos_to_ctrl(qpos: np.ndarray, ctrl_to_qpos_index: list[int | None]) -> np.ndarray:
    """Convert predicted qpos order to MuJoCo ctrl order for visualization files."""
    out = np.zeros((qpos.shape[0], len(ctrl_to_qpos_index)), dtype=np.float32)
    for ctrl_i, qpos_i in enumerate(ctrl_to_qpos_index):
        if qpos_i is not None:
            out[:, ctrl_i] = qpos[:, qpos_i]
    return out


def latency_ms_per_bin(predict_fn, total_bins: int, repeats: int = 3) -> float:
    """Estimate inference wall time normalized per decoded bin."""
    predict_fn()  # untimed warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times) / max(1, int(total_bins)))


def split_pretty_name(split_mode: str) -> str:
    """Make split names readable for plots."""
    return {
        "repeat_holdout": "repeat holdout",
        "gesture_holdout": "handle holdout",
        "session_holdout": "session holdout",
        "block_holdout": "block holdout",
    }.get(str(split_mode), str(split_mode))


