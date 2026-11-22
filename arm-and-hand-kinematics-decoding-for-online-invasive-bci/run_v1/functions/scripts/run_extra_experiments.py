#!/usr/bin/env python3
"""Additional classical-decoder experiments kept for VM-suite compatibility.

The main new-only training grid imports affine/Kalman helpers from here so that
classical baselines are evaluated with the same data loading conventions.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder_core.data import (
    DT,
    FeatureConfig,
    Trial,
    build_features_for_trial,
    default_data_root,
    load_dataset,
    resolve_channels,
    split_trials,
)
from decoder_core.metrics import mean_corr, r2_score, rmse, summarize
from decoder_core.models import Standardizer
from decoder_core.template_policy import build_template_features


HAND_DIMS = np.arange(0, 25, dtype=np.int64)
ARM_DIMS = np.arange(25, 32, dtype=np.int64)


@dataclass(slots=True)
class RidgeModel:
    x_scaler: Standardizer
    y_scaler: Standardizer
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = self.x_scaler.transform(np.asarray(x, dtype=np.float32))
        ys = xs @ self.weights + self.bias
        return self.y_scaler.inverse(ys).astype(np.float32)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    x_scaler = Standardizer.fit(x)
    y_scaler = Standardizer.fit(y)
    xs = x_scaler.transform(x).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    gram = xb.T @ xb
    reg = np.eye(gram.shape[0], dtype=np.float32) * np.float32(alpha)
    reg[-1, -1] = 0.0
    wb = np.linalg.solve(gram + reg, xb.T @ ys)
    return RidgeModel(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        weights=wb[:-1].astype(np.float32),
        bias=wb[-1].astype(np.float32),
    )


def clean_float(value: float | int | np.floating | np.integer) -> float | int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    out = float(value)
    return out if np.isfinite(out) else None


def clean_dict(d: dict) -> dict:
    return {k: clean_float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in d.items()}


def feature_grid() -> list[FeatureConfig]:
    return [
        FeatureConfig(channels="all", smooth_alpha=None, history=1, lag=0),
        FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0),
        FeatureConfig(channels="all", smooth_alpha=0.15, history=3, lag=0),
        FeatureConfig(channels="motor", smooth_alpha=0.15, history=3, lag=0),
    ]


def build_neural_matrix(trials: list[Trial], channels: np.ndarray, cfg: FeatureConfig, *, stop_before_last: bool) -> np.ndarray:
    xs = []
    for trial in trials:
        x = build_features_for_trial(trial, channels, cfg)
        if stop_before_last:
            x = x[:-1]
        xs.append(x)
    return np.concatenate(xs, axis=0).astype(np.float32)


def build_qpos_target(trials: list[Trial], dims: np.ndarray | None = None) -> np.ndarray:
    if dims is None:
        return np.concatenate([trial.qpos for trial in trials], axis=0).astype(np.float32)
    return np.concatenate([trial.qpos[:, dims] for trial in trials], axis=0).astype(np.float32)


def build_delta_target(trials: list[Trial], dims: np.ndarray | None = None) -> np.ndarray:
    if dims is None:
        return np.concatenate([trial.qpos[1:] - trial.qpos[:-1] for trial in trials], axis=0).astype(np.float32)
    return np.concatenate([trial.qpos[1:, dims] - trial.qpos[:-1, dims] for trial in trials], axis=0).astype(np.float32)


def build_state_delta_matrix(
    trials: list[Trial], channels: np.ndarray, cfg: FeatureConfig, dims: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for trial in trials:
        neural = build_features_for_trial(trial, channels, cfg)[:-1]
        state = np.concatenate([trial.qpos[:-1], trial.qvel[:-1]], axis=1)
        xs.append(np.concatenate([neural, state], axis=1))
        target = trial.qpos[1:] - trial.qpos[:-1]
        ys.append(target if dims is None else target[:, dims])
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def eval_direct_qpos(
    model: RidgeModel | tuple[RidgeModel, RidgeModel],
    trials: list[Trial],
    channels: np.ndarray,
    cfg: FeatureConfig,
    *,
    split: bool,
) -> dict:
    preds, trues = [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, cfg)
        if split:
            hand_model, arm_model = model  # type: ignore[misc]
            pred = np.empty_like(trial.qpos)
            pred[:, HAND_DIMS] = hand_model.predict(x)
            pred[:, ARM_DIMS] = arm_model.predict(x)
        else:
            pred = model.predict(x)  # type: ignore[union-attr]
        preds.append(pred)
        trues.append(trial.qpos)
    return summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))


def rollout_delta(
    model: RidgeModel,
    trials: list[Trial],
    channels: np.ndarray,
    cfg: FeatureConfig,
    *,
    state_conditioned: bool,
    gain: float,
    output_alpha: float,
    clip_abs: np.ndarray | None,
) -> dict:
    preds, trues = [], []
    for trial in trials:
        neural = build_features_for_trial(trial, channels, cfg)
        qpos = trial.qpos[0].astype(np.float32).copy()
        qvel = trial.qvel[0].astype(np.float32).copy()
        prev_delta = np.zeros(32, dtype=np.float32)
        out = []
        for t in range(trial.n_time - 1):
            if state_conditioned:
                x = np.concatenate([neural[t], qpos, qvel], axis=0)[None, :]
            else:
                x = neural[t : t + 1]
            delta = model.predict(x)[0] * np.float32(gain)
            if clip_abs is not None:
                delta = np.clip(delta, -clip_abs, clip_abs)
            delta = np.float32(output_alpha) * delta + np.float32(1.0 - output_alpha) * prev_delta
            next_qpos = qpos + delta
            qvel = delta / np.float32(DT)
            qpos = next_qpos.astype(np.float32)
            prev_delta = delta.astype(np.float32)
            out.append(qpos.copy())
        preds.append(np.asarray(out, dtype=np.float32))
        trues.append(trial.qpos[1:])
    return summarize(np.concatenate(trues, axis=0), np.concatenate(preds, axis=0))


def run_delta_and_split_experiments(dataset, train: list[Trial], val: list[Trial], test: list[Trial]) -> tuple[pd.DataFrame, dict]:
    rows = []
    ridge_alphas = [10.0, 100.0, 1000.0]

    for cfg in feature_grid():
        channels = resolve_channels(dataset, cfg)
        x_train = build_neural_matrix(train, channels, cfg, stop_before_last=False)

        for alpha in ridge_alphas:
            t0 = time.perf_counter()
            qpos_model = fit_ridge(x_train, build_qpos_target(train), alpha)
            val_metrics = eval_direct_qpos(qpos_model, val, channels, cfg, split=False)
            test_metrics = eval_direct_qpos(qpos_model, test, channels, cfg, split=False)
            rows.append(
                {
                    "experiment": "direct_qpos_all",
                    "feature": cfg.label(),
                    "ridge_alpha": alpha,
                    "train_ms": (time.perf_counter() - t0) * 1000.0,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

            t0 = time.perf_counter()
            hand_model = fit_ridge(x_train, build_qpos_target(train, HAND_DIMS), alpha)
            arm_model = fit_ridge(x_train, build_qpos_target(train, ARM_DIMS), alpha)
            val_metrics = eval_direct_qpos((hand_model, arm_model), val, channels, cfg, split=True)
            test_metrics = eval_direct_qpos((hand_model, arm_model), test, channels, cfg, split=True)
            rows.append(
                {
                    "experiment": "direct_qpos_split_hand_arm",
                    "feature": cfg.label(),
                    "ridge_alpha": alpha,
                    "train_ms": (time.perf_counter() - t0) * 1000.0,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

            x_delta = build_neural_matrix(train, channels, cfg, stop_before_last=True)
            y_delta = build_delta_target(train)
            delta_model = fit_ridge(x_delta, y_delta, alpha)
            clip_abs = np.percentile(np.abs(y_delta), 99.5, axis=0).astype(np.float32) * 2.0
            best_val = None
            best_score = -np.inf
            for gain in [0.5, 1.0, 1.5]:
                for output_alpha in [1.0, 0.65]:
                    val_metrics = rollout_delta(
                        delta_model,
                        val,
                        channels,
                        cfg,
                        state_conditioned=False,
                        gain=gain,
                        output_alpha=output_alpha,
                        clip_abs=clip_abs,
                    )
                    score = val_metrics["R2"] - 0.02 * val_metrics["jitter"]
                    if score > best_score:
                        best_score = score
                        best_val = (gain, output_alpha, val_metrics)
            assert best_val is not None
            gain, output_alpha, val_metrics = best_val
            test_metrics = rollout_delta(
                delta_model,
                test,
                channels,
                cfg,
                state_conditioned=False,
                gain=gain,
                output_alpha=output_alpha,
                clip_abs=clip_abs,
            )
            rows.append(
                {
                    "experiment": "qpos_delta_neural_rollout",
                    "feature": cfg.label(),
                    "ridge_alpha": alpha,
                    "gain": gain,
                    "output_alpha": output_alpha,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

            x_state, y_state = build_state_delta_matrix(train, channels, cfg)
            state_model = fit_ridge(x_state, y_state, alpha)
            clip_abs = np.percentile(np.abs(y_state), 99.5, axis=0).astype(np.float32) * 2.0
            best_val = None
            best_score = -np.inf
            for gain in [0.5, 1.0, 1.5]:
                for output_alpha in [1.0, 0.65]:
                    val_metrics = rollout_delta(
                        state_model,
                        val,
                        channels,
                        cfg,
                        state_conditioned=True,
                        gain=gain,
                        output_alpha=output_alpha,
                        clip_abs=clip_abs,
                    )
                    score = val_metrics["R2"] - 0.02 * val_metrics["jitter"]
                    if score > best_score:
                        best_score = score
                        best_val = (gain, output_alpha, val_metrics)
            assert best_val is not None
            gain, output_alpha, val_metrics = best_val
            test_metrics = rollout_delta(
                state_model,
                test,
                channels,
                cfg,
                state_conditioned=True,
                gain=gain,
                output_alpha=output_alpha,
                clip_abs=clip_abs,
            )
            rows.append(
                {
                    "experiment": "qpos_delta_state_rollout",
                    "feature": cfg.label(),
                    "ridge_alpha": alpha,
                    "gain": gain,
                    "output_alpha": output_alpha,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

    table = pd.DataFrame(rows)
    best = {}
    for name, group in table.groupby("experiment"):
        row = group.sort_values(["val_R2", "val_CC"], ascending=False).iloc[0].to_dict()
        best[name] = clean_dict(row)
    return table, best


def trial_phase(trial: Trial) -> np.ndarray:
    return np.linspace(0.0, 1.0, trial.n_time, dtype=np.float32)


def object_id(gesture: int, mapping: str) -> int:
    if mapping == "mod6":
        return gesture % 6
    if mapping == "block3":
        return gesture // 3
    raise ValueError(mapping)


def phase_object_features(trials: list[Trial], dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels = np.arange(256, dtype=np.int64)
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    xs, phase, gesture, length_phase_velocity = [], [], [], []
    for trial in trials:
        x = build_template_features(trial, channels, cfg, use_running_mean=True)
        p = trial_phase(trial)
        xs.append(x)
        phase.append(p[:, None])
        gesture.append(np.full((trial.n_time, 1), trial.gesture, dtype=np.int64))
        length_phase_velocity.append(np.full((trial.n_time, 1), 1.0 / max(1, trial.n_time - 1), dtype=np.float32))
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(phase, axis=0).astype(np.float32),
        np.concatenate(gesture, axis=0).astype(np.int64).ravel(),
        np.concatenate(length_phase_velocity, axis=0).astype(np.float32),
    )


def phase_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1, 1)
    return {
        "phase_R2": r2_score(y_true, y_pred),
        "phase_RMSE": rmse(y_true, y_pred),
        "phase_CC": mean_corr(y_true, y_pred),
        "phase_MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


def run_ridge_phase_object(train: list[Trial], test: list[Trial], mapping: str, gesture_limit: int) -> dict:
    train = [t for t in train if t.gesture < gesture_limit]
    test = [t for t in test if t.gesture < gesture_limit]
    x_train, phase_train, gesture_train, _ = phase_object_features(train, None)
    x_test, phase_test, gesture_test, _ = phase_object_features(test, None)
    obj_train = np.asarray([object_id(int(g), mapping) for g in gesture_train], dtype=np.int64)
    obj_test = np.asarray([object_id(int(g), mapping) for g in gesture_test], dtype=np.int64)
    n_obj = int(max(obj_train.max(initial=0), obj_test.max(initial=0)) + 1)
    phase_model = fit_ridge(x_train, phase_train, alpha=1000.0)
    y_onehot = np.zeros((obj_train.size, n_obj), dtype=np.float32)
    y_onehot[np.arange(obj_train.size), obj_train] = 1.0
    obj_model = fit_ridge(x_train, y_onehot, alpha=1000.0)
    phase_pred = np.clip(phase_model.predict(x_test)[:, 0], 0.0, 1.0)
    obj_score = obj_model.predict(x_test)
    obj_pred = np.argmax(obj_score, axis=1)
    return {
        "model": "ridge_phase_plus_object",
        "mapping": mapping,
        "gesture_set": f"first_{gesture_limit}",
        "n_objects": n_obj,
        "object_acc": float(np.mean(obj_pred == obj_test)),
        **phase_metrics(phase_test[:, 0], phase_pred),
    }


@dataclass(slots=True)
class AffineKalman:
    a: np.ndarray
    b: np.ndarray
    h: np.ndarray
    d: np.ndarray
    w: np.ndarray
    r_diag: np.ndarray
    x0: np.ndarray
    p0: np.ndarray
    htrh: np.ndarray


def fit_affine_kalman(state_trials: list[np.ndarray], obs_trials: list[np.ndarray], *, x0: np.ndarray) -> AffineKalman:
    x_all = np.concatenate(state_trials, axis=0).astype(np.float64)
    z_all = np.concatenate(obs_trials, axis=0).astype(np.float64)
    x1 = np.concatenate([x[:-1] for x in state_trials], axis=0).astype(np.float64)
    x2 = np.concatenate([x[1:] for x in state_trials], axis=0).astype(np.float64)
    xb = np.concatenate([x1, np.ones((x1.shape[0], 1))], axis=1)
    ab = np.linalg.pinv(xb) @ x2
    a = ab[:-1].T
    b = ab[-1]
    xobsb = np.concatenate([x_all, np.ones((x_all.shape[0], 1))], axis=1)
    hb = np.linalg.pinv(xobsb) @ z_all
    h = hb[:-1].T
    d = hb[-1]
    dyn_resid = x2 - (x1 @ a.T + b)
    obs_resid = z_all - (x_all @ h.T + d)
    w = np.cov(dyn_resid.T) if dyn_resid.shape[0] > 1 else np.eye(x0.size) * 1e-4
    if w.ndim == 0:
        w = np.asarray([[float(w)]])
    w = np.asarray(w, dtype=np.float64) + np.eye(x0.size) * 1e-5
    r_diag = np.var(obs_resid, axis=0) + 1e-2
    p0 = np.eye(x0.size, dtype=np.float64) * 0.05
    r_inv = 1.0 / r_diag
    htrh = h.T @ (h * r_inv[:, None])
    return AffineKalman(
        a=a.astype(np.float32),
        b=b.astype(np.float32),
        h=h.astype(np.float32),
        d=d.astype(np.float32),
        w=w.astype(np.float32),
        r_diag=r_diag.astype(np.float32),
        x0=x0.astype(np.float32),
        p0=p0.astype(np.float32),
        htrh=htrh.astype(np.float32),
    )


def run_affine_kalman(model: AffineKalman, obs: np.ndarray) -> np.ndarray:
    x = model.x0.astype(np.float32).copy()
    p = model.p0.astype(np.float32).copy()
    out = np.empty((obs.shape[0], x.size), dtype=np.float32)
    r_inv = (1.0 / model.r_diag).astype(np.float32)
    eye = np.eye(x.size, dtype=np.float32)
    for t in range(obs.shape[0]):
        x_pred = model.a @ x + model.b
        p_pred = model.a @ p @ model.a.T + model.w
        resid = obs[t] - (model.h @ x_pred + model.d)
        rhs = model.h.T @ (r_inv * resid)
        post_prec = np.linalg.pinv(p_pred) + model.htrh
        p = np.linalg.solve(post_prec, eye).astype(np.float32)
        x = x_pred + p @ rhs
        out[t] = x
    return out


def kalman_obs_for_trial(trial: Trial, channels: np.ndarray) -> np.ndarray:
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    x = build_template_features(trial, channels, cfg, use_running_mean=True)
    scaler = Standardizer.fit(x)
    return scaler.transform(x).astype(np.float32)


def build_phase_state_trial(trial: Trial) -> np.ndarray:
    p = trial_phase(trial)
    v = np.full_like(p, 1.0 / max(1, trial.n_time - 1))
    return np.stack([p, v], axis=1).astype(np.float32)


def run_kalman_phase_object(train: list[Trial], test: list[Trial], mapping: str, gesture_limit: int) -> tuple[list[dict], list[dict]]:
    train = [t for t in train if t.gesture < gesture_limit]
    test = [t for t in test if t.gesture < gesture_limit]
    channels = np.arange(256, dtype=np.int64)

    # Use one common scaler for Kalman observations, otherwise per-trial scaling leaks test distribution.
    train_obs_raw = [
        build_template_features(
            t,
            channels,
            FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0),
            use_running_mean=True,
        )
        for t in train
    ]
    obs_scaler = Standardizer.fit(np.concatenate(train_obs_raw, axis=0).astype(np.float32))
    train_obs = [obs_scaler.transform(x).astype(np.float32) for x in train_obs_raw]
    test_obs = [
        obs_scaler.transform(
            build_template_features(
                t,
                channels,
                FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0),
                use_running_mean=True,
            )
        ).astype(np.float32)
        for t in test
    ]

    n_obj = max(object_id(t.gesture, mapping) for t in train + test) + 1
    state_trials = []
    for trial in train:
        base = build_phase_state_trial(trial)
        onehot = np.zeros((trial.n_time, n_obj), dtype=np.float32)
        onehot[:, object_id(trial.gesture, mapping)] = 1.0
        state_trials.append(np.concatenate([base, onehot], axis=1))
    x0 = np.zeros(2 + n_obj, dtype=np.float32)
    x0[1] = np.median(np.concatenate([s[:, 1] for s in state_trials]))
    x0[2:] = 1.0 / n_obj
    global_model = fit_affine_kalman(state_trials, train_obs, x0=x0)
    pred_phase, true_phase, pred_obj, true_obj = [], [], [], []
    for trial, obs in zip(test, test_obs, strict=True):
        state_hat = run_affine_kalman(global_model, obs)
        pred_phase.append(np.clip(state_hat[:, 0], 0.0, 1.0))
        true_phase.append(trial_phase(trial))
        pred_obj.append(np.argmax(state_hat[:, 2:], axis=1))
        true_obj.append(np.full(trial.n_time, object_id(trial.gesture, mapping), dtype=np.int64))
    global_row = {
        "model": "affine_kalman_phase_plus_object_onehot",
        "mapping": mapping,
        "gesture_set": f"first_{gesture_limit}",
        "n_objects": n_obj,
        "object_acc": float(np.mean(np.concatenate(pred_obj) == np.concatenate(true_obj))),
        **phase_metrics(np.concatenate(true_phase), np.concatenate(pred_phase)),
    }

    per_object_rows = []
    per_object_pred_phase, per_object_true_phase = [], []
    for obj in range(n_obj):
        obj_train_pairs = [(t, o) for t, o in zip(train, train_obs, strict=True) if object_id(t.gesture, mapping) == obj]
        obj_test_pairs = [(t, o) for t, o in zip(test, test_obs, strict=True) if object_id(t.gesture, mapping) == obj]
        if not obj_train_pairs or not obj_test_pairs:
            continue
        states = [build_phase_state_trial(t) for t, _ in obj_train_pairs]
        obs_train = [o for _, o in obj_train_pairs]
        x0_phase = np.asarray([0.0, np.median(np.concatenate([s[:, 1] for s in states]))], dtype=np.float32)
        model = fit_affine_kalman(states, obs_train, x0=x0_phase)
        pp, tt = [], []
        for trial, obs in obj_test_pairs:
            state_hat = run_affine_kalman(model, obs)
            pp.append(np.clip(state_hat[:, 0], 0.0, 1.0))
            tt.append(trial_phase(trial))
        row = {
            "model": "affine_kalman_per_object_phase_known",
            "mapping": mapping,
            "gesture_set": f"first_{gesture_limit}",
            "object": obj,
            "n_test_trials": len(obj_test_pairs),
            **phase_metrics(np.concatenate(tt), np.concatenate(pp)),
        }
        per_object_rows.append(row)
        per_object_pred_phase.append(np.concatenate(pp))
        per_object_true_phase.append(np.concatenate(tt))

    pooled_row = {
        "model": "affine_kalman_per_object_phase_known_pooled",
        "mapping": mapping,
        "gesture_set": f"first_{gesture_limit}",
        "n_objects": n_obj,
        "object_acc": 1.0,
        **phase_metrics(np.concatenate(per_object_true_phase), np.concatenate(per_object_pred_phase)),
    }
    return [global_row, pooled_row], per_object_rows


def run_phase_object_experiments(train: list[Trial], test: list[Trial]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_object_rows = [], []
    for mapping in ["mod6", "block3"]:
        for gesture_limit in [18, 12, 6]:
            rows.append(run_ridge_phase_object(train, test, mapping, gesture_limit))
            kalman_rows, per_obj = run_kalman_phase_object(train, test, mapping, gesture_limit)
            rows.extend(kalman_rows)
            per_object_rows.extend(per_obj)
    return pd.DataFrame(rows), pd.DataFrame(per_object_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extra decoder experiments requested after the main policy.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to MM_sample_data_0116. Defaults to the workspace-relative data folder.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/extra_experiments"))
    args = parser.parse_args()

    data_root = args.data_root if args.data_root is not None else default_data_root(args.workspace)
    dataset = load_dataset(data_root)
    train, val, test = split_trials(dataset.trials)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("running qpos delta and split hand/arm experiments", flush=True)
    delta_table, delta_best = run_delta_and_split_experiments(dataset, train, val, test)
    delta_table.sort_values(["experiment", "val_R2", "val_CC"], ascending=[True, False, False]).to_csv(
        args.out_dir / "delta_split_results.csv", index=False
    )

    print("running phase/object ridge and Kalman experiments", flush=True)
    phase_table, per_object_table = run_phase_object_experiments(train, test)
    phase_table.to_csv(args.out_dir / "phase_object_results.csv", index=False)
    per_object_table.to_csv(args.out_dir / "phase_object_per_object_kalman.csv", index=False)

    summary = {
        "delta_split_best_by_experiment": delta_best,
        "phase_object_results": [clean_dict(row) for row in phase_table.to_dict(orient="records")],
        "notes": {
            "cc_definition": "Mean per-output Pearson correlation, skipping near-constant dimensions.",
            "hand_dims": HAND_DIMS.tolist(),
            "arm_dims": ARM_DIMS.tolist(),
            "object_mapping_warning": "No authoritative object labels were present in the files; mod6 and block3 are both provisional gesture-to-object mappings.",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
