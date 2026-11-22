from __future__ import annotations
"""Shared model utilities for linear and online-style decoders.

This module is intentionally small.  The large VM grid is defined in
``scripts/run_split_pca_model_suite.py``; this file provides reusable pieces
that the grid needs:

- ``Standardizer``: stores training-set mean/std for inputs and outputs.
- ``_fit_ridge_readout``: solves the ridge regression equation.
- ``Policy`` and ``OnlineController``: older streaming policy objects used by
  earlier online-style experiments.

The important idea is that scaling is always learned on training data and then
reused unchanged for validation, test, and online replay.  That makes the model
portable: the saved means/stds are part of the decoder, not something recomputed
from the test data.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import DT, FeatureConfig


@dataclass(slots=True)
class Standardizer:
    """Store mean/std so training, validation, test, and online data scale identically.

    In simple words: most models train better when every feature has roughly
    mean 0 and standard deviation 1.  ``fit`` learns those numbers on training
    rows only, ``transform`` applies them before the model, and ``inverse``
    converts model outputs back to physical qpos/qvel units.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        """Estimate feature-wise mean/std from training data only."""
        # x is a 2-D matrix: rows are time bins, columns are features.  For
        # example, x could be neural+phase features and y could be 32 qpos dims.
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        # Constant columns cannot be divided by zero; scale 1 leaves them as
        # zero after centering.
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Convert raw values to standardized values."""
        return (x - self.mean) / self.scale

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Convert standardized predictions back to original units."""
        return x * self.scale + self.mean


@dataclass(slots=True)
class Policy:
    """Serializable ridge/ELM policy used by older online-style experiments.

    This class is not the full VM leaderboard model grid.  It is the compact
    policy format used by earlier linear/ELM experiments and by the streaming
    ``OnlineController`` below.  It stores:

    - which neural channels/features were used,
    - how inputs and outputs were standardized,
    - the linear readout weights,
    - optional fixed random ELM hidden-layer weights,
    - control/smoothing parameters for online stepping.
    """

    model_kind: str
    target_mode: str
    feature_config: FeatureConfig
    channel_indices: np.ndarray
    x_scaler: Standardizer
    y_scaler: Standardizer
    ridge_alpha: float
    weights: np.ndarray
    bias: np.ndarray
    hidden_weights: np.ndarray | None = None
    hidden_bias: np.ndarray | None = None
    control_kp: float = 0.0
    control_vel_gain: float = 1.0
    control_max_speed: float | None = None
    output_alpha: float = 1.0
    dt: float = DT

    def _hidden(self, x_scaled: np.ndarray) -> np.ndarray:
        """Return either linear features or a fixed random ELM hidden layer."""
        if self.model_kind == "linear":
            # Linear/ridge policy: the scaled input is used directly.
            return x_scaled
        if self.model_kind == "elm":
            if self.hidden_weights is None or self.hidden_bias is None:
                raise ValueError("ELM policy is missing hidden weights.")
            # ELM means "extreme learning machine": the hidden layer is random
            # and fixed.  Only the final readout is fit with ridge regression.
            z = x_scaled @ self.hidden_weights + self.hidden_bias
            return np.maximum(z, 0.0)
        raise ValueError(f"Unknown model_kind: {self.model_kind}")

    def predict_target(self, x: np.ndarray) -> np.ndarray:
        """Predict the configured target from one or more feature rows."""
        # Input x arrives in real units.  It is standardized exactly like the
        # training data, mapped through the model, then converted back to target
        # units with y_scaler.inverse().
        x_scaled = self.x_scaler.transform(np.asarray(x, dtype=np.float32))
        h = self._hidden(x_scaled)
        y_scaled = h @ self.weights + self.bias
        return self.y_scaler.inverse(y_scaled).astype(np.float32)

    def save(self, path: Path | str) -> None:
        """Save policy arrays and metadata into a compressed NPZ file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Metadata records how the numbers should be interpreted.  Arrays are
        # saved separately so loading does not require pickle.
        metadata = {
            "model_kind": self.model_kind,
            "target_mode": self.target_mode,
            "feature_config": asdict(self.feature_config),
            "ridge_alpha": self.ridge_alpha,
            "control_kp": self.control_kp,
            "control_vel_gain": self.control_vel_gain,
            "control_max_speed": self.control_max_speed,
            "output_alpha": self.output_alpha,
            "dt": self.dt,
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata)),
            channel_indices=self.channel_indices.astype(np.int64),
            x_mean=self.x_scaler.mean,
            x_scale=self.x_scaler.scale,
            y_mean=self.y_scaler.mean,
            y_scale=self.y_scaler.scale,
            weights=self.weights,
            bias=self.bias,
            hidden_weights=np.asarray([]) if self.hidden_weights is None else self.hidden_weights,
            hidden_bias=np.asarray([]) if self.hidden_bias is None else self.hidden_bias,
        )

    @classmethod
    def load(cls, path: Path | str) -> "Policy":
        """Load a policy saved by Policy.save."""
        data = np.load(Path(path), allow_pickle=False)
        # The JSON metadata reconstructs the high-level choices; the NPZ arrays
        # reconstruct the exact fitted numerical decoder.
        metadata: dict[str, Any] = json.loads(str(data["metadata"]))
        fc = FeatureConfig(**metadata["feature_config"])
        hidden_weights = data["hidden_weights"]
        hidden_bias = data["hidden_bias"]
        return cls(
            model_kind=metadata["model_kind"],
            target_mode=metadata["target_mode"],
            feature_config=fc,
            channel_indices=data["channel_indices"],
            x_scaler=Standardizer(data["x_mean"], data["x_scale"]),
            y_scaler=Standardizer(data["y_mean"], data["y_scale"]),
            ridge_alpha=float(metadata["ridge_alpha"]),
            weights=data["weights"],
            bias=data["bias"],
            hidden_weights=None if hidden_weights.size == 0 else hidden_weights,
            hidden_bias=None if hidden_bias.size == 0 else hidden_bias,
            control_kp=float(metadata["control_kp"]),
            control_vel_gain=float(metadata["control_vel_gain"]),
            control_max_speed=metadata["control_max_speed"],
            output_alpha=float(metadata["output_alpha"]),
            dt=float(metadata["dt"]),
        )


def _fit_ridge_readout(h: np.ndarray, y_scaled: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ridge-regularized affine readout from features to scaled targets.

    This solves:

        y_scaled ~= h @ weights + bias

    ``alpha`` is the ridge penalty.  Larger alpha means smaller/smoother weights
    and less overfitting, but too much alpha can underfit.
    """
    h = np.asarray(h, dtype=np.float32)
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    # Add a column of ones so the bias can be solved in the same linear system
    # as the weights.
    ones = np.ones((h.shape[0], 1), dtype=np.float32)
    hb = np.concatenate([h, ones], axis=1)
    gram = hb.T @ hb
    reg = np.eye(gram.shape[0], dtype=np.float32) * np.float32(alpha)
    # Do not regularize the bias row/column.
    reg[-1, -1] = 0.0
    rhs = hb.T @ y_scaled
    try:
        wb = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        wb = np.linalg.pinv(gram + reg) @ rhs
    return wb[:-1].astype(np.float32), wb[-1].astype(np.float32)


def fit_policy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_kind: str,
    target_mode: str,
    feature_config: FeatureConfig,
    channel_indices: np.ndarray,
    ridge_alpha: float,
    hidden_dim: int = 256,
    seed: int = 0,
) -> Policy:
    """Fit an older linear/ELM policy with shared scaling and ridge readout.

    Inputs:
    - ``x``: feature matrix, usually neural history features.
    - ``y``: target matrix, usually qpos, qvel, or qpos+qvel depending on
      ``target_mode``.

    Output:
    - a saved ``Policy`` object that can be stepped online by
      ``OnlineController``.
    """
    # Fit scalers on training matrices only, then train in standardized units.
    x_scaler = Standardizer.fit(x)
    y_scaler = Standardizer.fit(y)
    xs = x_scaler.transform(x).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)

    hidden_weights = None
    hidden_bias = None
    if model_kind == "linear":
        # Pure ridge baseline: no nonlinear hidden layer.
        h = xs
    elif model_kind == "elm":
        # ELM baseline: fixed random ReLU expansion followed by ridge readout.
        # This gives a cheap nonlinear model without backpropagation.
        rng = np.random.default_rng(seed)
        hidden_weights = rng.normal(0.0, 1.0 / np.sqrt(max(1, xs.shape[1])), size=(xs.shape[1], hidden_dim)).astype(
            np.float32
        )
        hidden_bias = rng.normal(0.0, 0.1, size=(hidden_dim,)).astype(np.float32)
        h = np.maximum(xs @ hidden_weights + hidden_bias, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    weights, bias = _fit_ridge_readout(h, ys, ridge_alpha)
    return Policy(
        model_kind=model_kind,
        target_mode=target_mode,
        feature_config=feature_config,
        channel_indices=channel_indices,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge_alpha=ridge_alpha,
        weights=weights,
        bias=bias,
        hidden_weights=hidden_weights,
        hidden_bias=hidden_bias,
    )


class OnlineController:
    """Allocation-light streaming wrapper around a trained policy.

    This class shows how an older saved policy would run online: one 100 Hz
    neural bin enters ``step()``, the internal neural history buffer updates,
    and one 32-D qpos command comes out.  It is useful for BRAND-style thinking
    because it avoids seeing future bins.
    """

    def __init__(self, policy: Policy):
        self.policy = policy
        self.channels = policy.channel_indices
        self.history = int(policy.feature_config.history)
        self.n_channels = int(self.channels.size)
        # feature is the flattened history window passed to the decoder:
        # [current selected channels, previous selected channels, ...].
        self.feature = np.zeros((1, self.n_channels * self.history), dtype=np.float32)
        self.buffer = np.zeros((self.history, self.n_channels), dtype=np.float32)
        # smooth stores the causal EMA neural state for online use.
        self.smooth = np.zeros(self.n_channels, dtype=np.float32)
        # qpos is the controller's current command.  For qvel/PV policies,
        # predictions are integrated into this position command.
        self.qpos = np.zeros(32, dtype=np.float32)
        self.prev_output = np.zeros(32, dtype=np.float32)
        self.initialized = False

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Clear streaming state before a new online trial."""
        # Start each attempt with an empty neural history and a known arm pose.
        self.feature.fill(0.0)
        self.buffer.fill(0.0)
        self.smooth.fill(0.0)
        self.qpos[:] = 0.0 if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float32)
        self.prev_output[:] = self.qpos
        self.initialized = False

    def _observe(self, spikes_256: np.ndarray) -> np.ndarray:
        """Update causal neural smoothing/history buffer from one 256-channel bin."""
        # A real online loop would call this once per bin with the latest 256
        # Blackrock channels.  Only the channels selected during training are
        # kept, so the online input matches the training input exactly.
        obs = np.asarray(spikes_256, dtype=np.float32)[self.channels]
        alpha = self.policy.feature_config.smooth_alpha
        if alpha is None:
            # No neural smoothing requested.
            filt = obs
        elif not self.initialized:
            # Initialize the EMA with the first observation to avoid a startup
            # jump from zero.
            self.smooth[:] = obs
            filt = self.smooth
        else:
            # Causal exponential moving average: present bin + past state only.
            self.smooth[:] = np.float32(alpha) * obs + np.float32(1.0 - alpha) * self.smooth
            filt = self.smooth
        self.initialized = True
        # Shift older bins down the history window and place the newest filtered
        # observation at the front.
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = filt
        self.feature[0] = self.buffer.reshape(-1)
        return self.feature

    def step(self, spikes_256: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Decode one online bin and return the current qpos command."""
        x = self._observe(spikes_256)
        target = self.policy.predict_target(x)[0]

        if self.policy.target_mode == "qpos":
            # Direct position policy: the model already outputs qpos.
            raw_qpos = target[:32]
        elif self.policy.target_mode == "qvel":
            # Velocity policy: integrate qvel over one 100 Hz bin to get qpos.
            raw_qpos = self.qpos + np.float32(self.policy.dt) * target[:32]
        elif self.policy.target_mode == "pv":
            # Position+velocity policy: combine predicted velocity with a small
            # corrective pull toward predicted position.
            pred_pos = target[:32]
            pred_vel = target[32:64]
            qdot = self.policy.control_vel_gain * pred_vel + self.policy.control_kp * (pred_pos - self.qpos)
            if self.policy.control_max_speed is not None:
                qdot = np.clip(qdot, -self.policy.control_max_speed, self.policy.control_max_speed)
            raw_qpos = self.qpos + np.float32(self.policy.dt) * qdot
        else:
            raise ValueError(f"Unsupported online target mode: {self.policy.target_mode}")

        alpha = np.float32(self.policy.output_alpha)
        # Final output smoothing, again causal: new command blended with the
        # previous command.
        out = alpha * raw_qpos + (np.float32(1.0) - alpha) * self.prev_output
        self.qpos[:] = out
        self.prev_output[:] = out
        return out.copy(), {"target": target.copy()}
