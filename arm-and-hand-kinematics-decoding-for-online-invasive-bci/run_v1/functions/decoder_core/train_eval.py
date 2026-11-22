from __future__ import annotations
"""Older shared train/evaluate helpers for causal online-style policies.

These utilities support earlier scripts and keep the same data/metric
conventions as the current VM suite.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DT,
    Dataset,
    FeatureConfig,
    Trial,
    build_features_for_trial,
    build_matrix,
    default_data_root,
    load_dataset,
    resolve_channels,
    split_trials,
)
from .metrics import summarize
from .models import OnlineController, Policy, fit_policy


def simulate_policy(policy: Policy, trials: list[Trial], *, initial: str = "first") -> tuple[np.ndarray, np.ndarray]:
    controller = OnlineController(policy)
    preds, trues = [], []
    lag = int(policy.feature_config.lag)
    for trial in trials:
        initial_qpos = trial.qpos[0] if initial == "first" else np.zeros(32, dtype=np.float32)
        controller.reset(initial_qpos)
        n = trial.n_time - lag
        if n <= 0:
            continue
        trial_preds = []
        for t in range(n):
            qpos, _ = controller.step(trial.neural[t])
            trial_preds.append(qpos)
        preds.append(np.asarray(trial_preds, dtype=np.float32))
        trues.append(trial.qpos[lag : lag + n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def predict_targets_by_trial(policy: Policy, trials: list[Trial]) -> list[np.ndarray]:
    preds = []
    for trial in trials:
        x = build_features_for_trial(trial, policy.channel_indices, policy.feature_config)
        if policy.feature_config.lag:
            x = x[:-policy.feature_config.lag]
        preds.append(policy.predict_target(x))
    return preds


def simulate_pv_targets(
    policy: Policy,
    trials: list[Trial],
    targets: list[np.ndarray],
    *,
    kp: float,
    vel_gain: float,
    max_speed: float | None,
    output_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    lag = int(policy.feature_config.lag)
    pred_trials, true_trials = [], []
    for trial, target in zip(trials, targets, strict=True):
        qpos = trial.qpos[0].astype(np.float32).copy()
        prev = qpos.copy()
        out = np.empty((target.shape[0], 32), dtype=np.float32)
        for t in range(target.shape[0]):
            pred_pos = target[t, :32]
            pred_vel = target[t, 32:64]
            qdot = vel_gain * pred_vel + kp * (pred_pos - qpos)
            if max_speed is not None:
                qdot = np.clip(qdot, -max_speed, max_speed)
            raw = qpos + np.float32(policy.dt) * qdot
            qpos = np.float32(output_alpha) * raw + np.float32(1.0 - output_alpha) * prev
            prev = qpos
            out[t] = qpos
        pred_trials.append(out)
        true_trials.append(trial.qpos[lag : lag + target.shape[0]])
    return np.concatenate(true_trials, axis=0), np.concatenate(pred_trials, axis=0)


def direct_predict_policy(policy: Policy, dataset: Dataset, trials: list[Trial]) -> tuple[np.ndarray, np.ndarray]:
    from .data import aligned_target

    channels = resolve_channels(dataset, policy.feature_config)
    preds, trues = [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, policy.feature_config)
        if policy.feature_config.lag:
            x = x[:-policy.feature_config.lag]
        target = policy.predict_target(x)
        if policy.target_mode == "qpos":
            qpos = target[:, :32]
        elif policy.target_mode == "pv":
            qpos = target[:, :32]
        elif policy.target_mode == "qvel":
            qpos = np.zeros((target.shape[0], 32), dtype=np.float32)
            qpos[0] = trial.qpos[0]
            for t in range(1, target.shape[0]):
                qpos[t] = qpos[t - 1] + DT * target[t, :32]
        else:
            raise ValueError(policy.target_mode)
        y = aligned_target(trial, "qpos", policy.feature_config.lag)
        preds.append(qpos)
        trues.append(y)
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def tune_control(policy: Policy, val_trials: list[Trial]) -> dict[str, float | None]:
    if policy.target_mode != "pv":
        return {"control_kp": 0.0, "control_vel_gain": 1.0, "control_max_speed": None, "output_alpha": 1.0}
    targets = predict_targets_by_trial(policy, val_trials)
    grid = []
    for kp in (0.0, 3.0, 8.0, 15.0, 25.0):
        for vg in (0.0, 0.5, 1.0):
            for max_speed in (None, 8.0):
                for output_alpha in (1.0, 0.65):
                    grid.append((kp, vg, max_speed, output_alpha))
    best = None
    best_score = -np.inf
    for kp, vg, max_speed, output_alpha in grid:
        policy.control_kp = kp
        policy.control_vel_gain = vg
        policy.control_max_speed = max_speed
        policy.output_alpha = output_alpha
        y_true, y_pred = simulate_pv_targets(
            policy,
            val_trials,
            targets,
            kp=kp,
            vel_gain=vg,
            max_speed=max_speed,
            output_alpha=output_alpha,
        )
        m = summarize(y_true, y_pred)
        score = m["R2"] - 0.02 * m["jitter"]
        if score > best_score:
            best_score = score
            best = {
                "control_kp": kp,
                "control_vel_gain": vg,
                "control_max_speed": max_speed,
                "output_alpha": output_alpha,
            }
    assert best is not None
    policy.control_kp = float(best["control_kp"])
    policy.control_vel_gain = float(best["control_vel_gain"])
    policy.control_max_speed = best["control_max_speed"]
    policy.output_alpha = float(best["output_alpha"])
    return best


def latency_benchmark(policy: Policy, trial: Trial, steps: int = 3000) -> dict[str, float]:
    controller = OnlineController(policy)
    controller.reset(trial.qpos[0])
    spikes = trial.neural
    n = spikes.shape[0]
    t0 = time.perf_counter()
    for i in range(steps):
        controller.step(spikes[i % n])
    elapsed = time.perf_counter() - t0
    return {
        "steps": float(steps),
        "total_ms": elapsed * 1000.0,
        "step_ms": elapsed * 1000.0 / steps,
        "hz": steps / elapsed,
    }


def config_grid(profile: str) -> list[dict]:
    if profile == "smoke":
        feature_cfgs = [
            FeatureConfig(channels="motor", smooth_alpha=None, history=1, lag=0),
            FeatureConfig(channels="motor", smooth_alpha=0.15, history=3, lag=0),
        ]
        return [
            {"model_kind": "linear", "target_mode": "qpos", "ridge_alpha": 100.0, "feature_config": feature_cfgs[0]},
            {"model_kind": "linear", "target_mode": "pv", "ridge_alpha": 100.0, "feature_config": feature_cfgs[1]},
        ]

    channels = ["all", "motor", "dM1", "vM1"]
    alphas = [None, 0.15]
    histories = [1, 3, 5]
    lags = [0, 2]
    target_modes = ["qpos", "pv"]
    ridge_alphas = [10.0, 100.0, 1000.0]
    cfgs = []
    for ch in channels:
        for alpha in alphas:
            for hist in histories:
                for lag in lags:
                    fc = FeatureConfig(channels=ch, smooth_alpha=alpha, history=hist, lag=lag)
                    for target in target_modes:
                        for ridge in ridge_alphas:
                            cfgs.append(
                                {
                                    "model_kind": "linear",
                                    "target_mode": target,
                                    "ridge_alpha": ridge,
                                    "feature_config": fc,
                                }
                            )
    if profile == "quick":
        return cfgs[:]

    # Full profile adds random-feature readouts on a smaller, online-friendly subset.
    for ch in ["all", "motor", "dM1", "vM1", "no_pMC"]:
        for alpha in [None, 0.10, 0.20]:
            for hist in [1, 3, 5]:
                for lag in [0, 2, 5]:
                    for ridge in [10.0, 100.0, 1000.0]:
                        cfgs.append(
                            {
                                "model_kind": "linear",
                                "target_mode": "qvel",
                                "ridge_alpha": ridge,
                                "feature_config": FeatureConfig(channels=ch, smooth_alpha=alpha, history=hist, lag=lag),
                            }
                        )
    for ch in ["motor", "dM1", "all"]:
        for alpha in [None, 0.10, 0.20]:
            for hist in [1, 3]:
                for lag in [0, 2]:
                    for target in ["qpos", "pv"]:
                        for hidden in [128, 256, 512]:
                            cfgs.append(
                                {
                                    "model_kind": "elm",
                                    "target_mode": target,
                                    "ridge_alpha": 100.0,
                                    "hidden_dim": hidden,
                                    "feature_config": FeatureConfig(
                                        channels=ch, smooth_alpha=alpha, history=hist, lag=lag
                                    ),
                                }
                            )
    return cfgs


def train_one(dataset: Dataset, train: list[Trial], val: list[Trial], cfg: dict, seed: int) -> tuple[Policy, dict]:
    fc: FeatureConfig = cfg["feature_config"]
    channels = resolve_channels(dataset, fc)
    x_train, y_train = build_matrix(train, dataset, fc, cfg["target_mode"])
    t0 = time.perf_counter()
    policy = fit_policy(
        x_train,
        y_train,
        model_kind=cfg["model_kind"],
        target_mode=cfg["target_mode"],
        feature_config=fc,
        channel_indices=channels,
        ridge_alpha=float(cfg["ridge_alpha"]),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        seed=seed,
    )
    train_ms = (time.perf_counter() - t0) * 1000.0
    tune_control(policy, val)
    yv, pv = simulate_policy(policy, val)
    online = summarize(yv, pv)
    yd, pd_ = direct_predict_policy(policy, dataset, val)
    direct = summarize(yd, pd_)
    metric = {
        **online,
        "direct_R2": direct["R2"],
        "direct_RMSE": direct["RMSE"],
        "direct_CC": direct["CC"],
        "train_ms": train_ms,
        "n_features": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_channels": int(channels.size),
        "model_kind": cfg["model_kind"],
        "target_mode": cfg["target_mode"],
        "ridge_alpha": float(cfg["ridge_alpha"]),
        "hidden_dim": int(cfg.get("hidden_dim", 0)),
        "feature": fc.label(),
        "channels": fc.channels,
        "smooth_alpha": np.nan if fc.smooth_alpha is None else float(fc.smooth_alpha),
        "history": int(fc.history),
        "lag": int(fc.lag),
        "control_kp": float(policy.control_kp),
        "control_vel_gain": float(policy.control_vel_gain),
        "control_max_speed": np.nan if policy.control_max_speed is None else float(policy.control_max_speed),
        "output_alpha": float(policy.output_alpha),
    }
    return policy, metric


def fit_sensor_surrogate(train_trials: list[Trial], val_trials: list[Trial], alpha: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([t.qpos for t in train_trials + val_trials], axis=0).astype(np.float32)
    y = np.concatenate([t.sensor for t in train_trials + val_trials], axis=0).astype(np.float32)
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    gram = xb.T @ xb
    reg = np.eye(gram.shape[0], dtype=np.float32) * np.float32(alpha)
    reg[-1, -1] = 0.0
    wb = np.linalg.solve(gram + reg, xb.T @ y)
    return wb[:-1].astype(np.float32), wb[-1].astype(np.float32)


def export_sim_trial(
    path: Path,
    policy: Policy,
    trial: Trial,
    sensor_w: np.ndarray,
    sensor_b: np.ndarray,
    joint_names: list[str],
) -> None:
    y_true, y_pred = simulate_policy(policy, [trial])
    pred_sensor = y_pred @ sensor_w + sensor_b
    true_sensor = trial.sensor[policy.feature_config.lag : policy.feature_config.lag + y_true.shape[0]]
    payload = {
        "dt": policy.dt,
        "gesture": trial.gesture,
        "repeat": trial.repeat,
        "joint_names": joint_names,
        "true_qpos": np.round(y_true, 5).tolist(),
        "pred_qpos": np.round(y_pred, 5).tolist(),
        "true_sensor": np.round(true_sensor, 5).tolist(),
        "pred_sensor": np.round(pred_sensor, 5).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.ARMHAND_SIM_DATA = " + json.dumps(payload) + ";\n")


def run_training(
    *,
    workspace: Path,
    profile: str,
    artifact_dir: Path,
    max_configs: int | None = None,
    seed: int = 42,
    data_root: Path | str | None = None,
) -> dict:
    dataset = load_dataset(data_root if data_root is not None else default_data_root(workspace))
    train, val, test = split_trials(dataset.trials)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir = artifact_dir / "results"
    results_dir.mkdir(exist_ok=True)

    cfgs = config_grid(profile)
    if max_configs is not None:
        cfgs = cfgs[:max_configs]

    rows = []
    best_policy = None
    best_score = -np.inf
    best_idx = -1
    t_all = time.perf_counter()
    for i, cfg in enumerate(cfgs, 1):
        policy, metric = train_one(dataset, train, val, cfg, seed + i)
        metric["idx"] = i
        rows.append(metric)
        score = metric["R2"] + 0.15 * metric["direct_R2"] - 0.01 * metric["jitter"]
        if score > best_score:
            best_score = score
            best_policy = policy
            best_idx = i
        if i == 1 or i % 10 == 0 or i == len(cfgs):
            elapsed = time.perf_counter() - t_all
            print(
                f"[{i:04d}/{len(cfgs):04d}] best_idx={best_idx} "
                f"best_score={best_score:+.4f} last_online_R2={metric['R2']:+.4f} "
                f"last_direct_R2={metric['direct_R2']:+.4f} elapsed={elapsed/60:.1f}m",
                flush=True,
            )
        if i % 25 == 0:
            pd.DataFrame(rows).sort_values("R2", ascending=False).to_csv(
                results_dir / "leaderboard_partial.csv", index=False
            )

    if best_policy is None:
        raise RuntimeError("No policy was trained.")

    leaderboard = pd.DataFrame(rows)
    leaderboard.sort_values(["R2", "direct_R2"], ascending=False).to_csv(results_dir / "leaderboard.csv", index=False)

    y_test, p_test = simulate_policy(best_policy, test)
    test_metrics = summarize(y_test, p_test)
    y_direct, p_direct = direct_predict_policy(best_policy, dataset, test)
    direct_metrics = summarize(y_direct, p_direct)
    lat = latency_benchmark(best_policy, test[0])
    best_policy.save(artifact_dir / "best_policy.npz")
    stable_artifact_dir = workspace / "artifacts"
    stable_artifact_dir.mkdir(parents=True, exist_ok=True)
    best_policy.save(stable_artifact_dir / "best_policy.npz")

    sensor_w, sensor_b = fit_sensor_surrogate(train, val)
    np.savez_compressed(artifact_dir / "sensor_surrogate.npz", weights=sensor_w, bias=sensor_b)
    np.savez_compressed(stable_artifact_dir / "sensor_surrogate.npz", weights=sensor_w, bias=sensor_b)
    export_sim_trial(artifact_dir / "sim_trial.js", best_policy, test[0], sensor_w, sensor_b, dataset.joint_names)
    export_sim_trial(stable_artifact_dir / "sim_trial.js", best_policy, test[0], sensor_w, sensor_b, dataset.joint_names)

    top = leaderboard.sort_values(["R2", "direct_R2"], ascending=False).head(20)
    top.to_csv(results_dir / "top20.csv", index=False)
    summary = {
        "profile": profile,
        "n_configs": len(cfgs),
        "best_idx": best_idx,
        "best_validation_online": rows[best_idx - 1],
        "test_online": test_metrics,
        "test_direct": {
            "R2": direct_metrics["R2"],
            "RMSE": direct_metrics["RMSE"],
            "CC": direct_metrics["CC"],
            "med_R2": direct_metrics["med_R2"],
            "worst_R2": direct_metrics["worst_R2"],
        },
        "latency": lat,
        "policy_file": str(artifact_dir / "best_policy.npz"),
        "sim_file": str(artifact_dir / "sim_trial.js"),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (stable_artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
