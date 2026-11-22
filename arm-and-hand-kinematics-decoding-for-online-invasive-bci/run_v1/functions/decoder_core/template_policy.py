from __future__ import annotations
"""Template-policy utilities from the earlier task-assisted decoder work.

The new split/PCA VM suite has its own hybrid template bank, but this module is
kept because several compatibility scripts still import its template-feature and
Ridge helper logic.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    Dataset,
    FeatureConfig,
    Trial,
    build_features_for_trial,
    default_data_root,
    load_dataset,
    resolve_channels,
    split_trials,
)
from .metrics import summarize
from .models import Standardizer, _fit_ridge_readout


@dataclass(slots=True)
class TemplatePolicy:
    feature_config: FeatureConfig
    channel_indices: np.ndarray
    x_scaler: Standardizer
    class_weights: np.ndarray
    class_bias: np.ndarray
    templates_qpos: list[np.ndarray]
    templates_sensor: list[np.ndarray]
    prob_alpha: float = 0.15
    temperature: float = 0.35
    output_alpha: float = 0.85
    use_running_mean: bool = False

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "feature_config": asdict(self.feature_config),
            "prob_alpha": self.prob_alpha,
            "temperature": self.temperature,
            "output_alpha": self.output_alpha,
            "n_templates": len(self.templates_qpos),
            "use_running_mean": self.use_running_mean,
        }
        arrays = {
            "metadata": np.asarray(json.dumps(metadata)),
            "channel_indices": self.channel_indices.astype(np.int64),
            "x_mean": self.x_scaler.mean,
            "x_scale": self.x_scaler.scale,
            "class_weights": self.class_weights,
            "class_bias": self.class_bias,
        }
        for i, q in enumerate(self.templates_qpos):
            arrays[f"template_qpos_{i}"] = q.astype(np.float32)
        for i, s in enumerate(self.templates_sensor):
            arrays[f"template_sensor_{i}"] = s.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "TemplatePolicy":
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        n = int(metadata["n_templates"])
        return cls(
            feature_config=FeatureConfig(**metadata["feature_config"]),
            channel_indices=data["channel_indices"],
            x_scaler=Standardizer(data["x_mean"], data["x_scale"]),
            class_weights=data["class_weights"],
            class_bias=data["class_bias"],
            templates_qpos=[data[f"template_qpos_{i}"] for i in range(n)],
            templates_sensor=[data[f"template_sensor_{i}"] for i in range(n)],
            prob_alpha=float(metadata["prob_alpha"]),
            temperature=float(metadata["temperature"]),
            output_alpha=float(metadata["output_alpha"]),
            use_running_mean=bool(metadata.get("use_running_mean", False)),
        )


class OnlineTemplateController:
    def __init__(self, policy: TemplatePolicy):
        self.policy = policy
        self.channels = policy.channel_indices
        self.history = int(policy.feature_config.history)
        self.n_channels = int(self.channels.size)
        self.buffer = np.zeros((self.history, self.n_channels), dtype=np.float32)
        self.feature = np.zeros((1, self.history * self.n_channels), dtype=np.float32)
        self.smooth = np.zeros(self.n_channels, dtype=np.float32)
        self.running_sum = np.zeros(self.n_channels, dtype=np.float32)
        feature_dim = self.history * self.n_channels + (self.n_channels if policy.use_running_mean else 0)
        self.model_feature = np.zeros((1, feature_dim), dtype=np.float32)
        self.prob = np.full(len(policy.templates_qpos), 1.0 / len(policy.templates_qpos), dtype=np.float32)
        self.prev_qpos = np.zeros(32, dtype=np.float32)
        self.t = 0
        self.initialized = False

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        self.buffer.fill(0.0)
        self.feature.fill(0.0)
        self.smooth.fill(0.0)
        self.running_sum.fill(0.0)
        self.prob.fill(1.0 / len(self.prob))
        self.prev_qpos[:] = 0.0 if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float32)
        self.t = 0
        self.initialized = False

    def _softmax(self, score: np.ndarray) -> np.ndarray:
        temp = max(np.float32(self.policy.temperature), np.float32(1e-3))
        z = score / temp
        z = z - np.max(z)
        e = np.exp(z).astype(np.float32)
        return e / np.sum(e)

    def _observe(self, spikes_256: np.ndarray) -> np.ndarray:
        obs = np.asarray(spikes_256, dtype=np.float32)[self.channels]
        alpha = self.policy.feature_config.smooth_alpha
        if alpha is None:
            filt = obs
        elif not self.initialized:
            self.smooth[:] = obs
            filt = self.smooth
        else:
            self.smooth[:] = np.float32(alpha) * obs + np.float32(1.0 - alpha) * self.smooth
            filt = self.smooth
        self.initialized = True
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = filt
        self.feature[0] = self.buffer.reshape(-1)
        if not self.policy.use_running_mean:
            return self.feature
        self.running_sum[:] += filt
        running_mean = self.running_sum / np.float32(self.t + 1)
        self.model_feature[0, : self.feature.shape[1]] = self.feature[0]
        self.model_feature[0, self.feature.shape[1] :] = running_mean
        return self.model_feature

    def step(self, spikes_256: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
        x = self._observe(spikes_256)
        xs = self.policy.x_scaler.transform(x)
        score = xs @ self.policy.class_weights + self.policy.class_bias
        p_now = self._softmax(score[0])
        pa = np.float32(self.policy.prob_alpha)
        self.prob[:] = (np.float32(1.0) - pa) * self.prob + pa * p_now
        self.prob[:] = self.prob / np.sum(self.prob)

        q = np.zeros(32, dtype=np.float32)
        for g, template in enumerate(self.policy.templates_qpos):
            idx = min(self.t, template.shape[0] - 1)
            q += self.prob[g] * template[idx]
        oa = np.float32(self.policy.output_alpha)
        out = oa * q + (np.float32(1.0) - oa) * self.prev_qpos
        self.prev_qpos[:] = out
        self.t += 1
        return out.copy(), {"prob": self.prob.copy(), "gesture": int(np.argmax(self.prob))}


def build_template_features(
    trial: Trial,
    channels: np.ndarray,
    cfg: FeatureConfig,
    *,
    use_running_mean: bool,
) -> np.ndarray:
    x = build_features_for_trial(trial, channels, cfg)
    if not use_running_mean:
        return x
    from .data import causal_ema

    neural = causal_ema(trial.neural[:, channels], cfg.smooth_alpha)
    running = np.cumsum(neural, axis=0) / np.arange(1, neural.shape[0] + 1, dtype=np.float32)[:, None]
    return np.concatenate([x, running.astype(np.float32)], axis=1)


def fit_template_policy(
    dataset: Dataset,
    train: list[Trial],
    cfg: FeatureConfig,
    ridge_alpha: float,
    *,
    use_running_mean: bool = False,
) -> TemplatePolicy:
    channels = resolve_channels(dataset, cfg)
    xs, labels = [], []
    for trial in train:
        x = build_template_features(trial, channels, cfg, use_running_mean=use_running_mean)
        if cfg.lag:
            x = x[:-cfg.lag]
        xs.append(x)
        labels.append(np.full(x.shape[0], trial.gesture, dtype=np.int64))
    x_train = np.concatenate(xs, axis=0)
    y_idx = np.concatenate(labels, axis=0)
    x_scaler = Standardizer.fit(x_train)
    xs_scaled = x_scaler.transform(x_train).astype(np.float32)
    n_g = max(t.gesture for t in train) + 1
    y = np.zeros((x_train.shape[0], n_g), dtype=np.float32)
    y[np.arange(y.shape[0]), y_idx] = 1.0
    class_weights, class_bias = _fit_ridge_readout(xs_scaled, y, ridge_alpha)

    templates_qpos = []
    templates_sensor = []
    for gesture in range(n_g):
        candidates = [t for t in train if t.gesture == gesture]
        if not candidates:
            raise ValueError(f"No template source for gesture {gesture}")
        # Kinematic template is identical across repeats in this dataset; use first train repeat.
        templates_qpos.append(candidates[0].qpos.copy())
        templates_sensor.append(candidates[0].sensor.copy())

    return TemplatePolicy(
        feature_config=cfg,
        channel_indices=channels,
        x_scaler=x_scaler,
        class_weights=class_weights,
        class_bias=class_bias,
        templates_qpos=templates_qpos,
        templates_sensor=templates_sensor,
        use_running_mean=use_running_mean,
    )


def simulate_template(policy: TemplatePolicy, trials: list[Trial]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl = OnlineTemplateController(policy)
    preds, trues, gestures = [], [], []
    lag = int(policy.feature_config.lag)
    for trial in trials:
        ctrl.reset(trial.qpos[0])
        n = trial.n_time - lag
        p_trial = []
        g_trial = []
        for t in range(n):
            q, info = ctrl.step(trial.neural[t])
            p_trial.append(q)
            g_trial.append(info["gesture"])
        preds.append(np.asarray(p_trial, dtype=np.float32))
        trues.append(trial.qpos[lag : lag + n])
        gestures.append(np.asarray(g_trial, dtype=np.int64))
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0), np.concatenate(gestures, axis=0)


def template_latency(policy: TemplatePolicy, trial: Trial, steps: int = 3000) -> dict[str, float]:
    ctrl = OnlineTemplateController(policy)
    ctrl.reset(trial.qpos[0])
    t0 = time.perf_counter()
    for i in range(steps):
        ctrl.step(trial.neural[i % trial.n_time])
    elapsed = time.perf_counter() - t0
    return {"steps": float(steps), "total_ms": elapsed * 1000.0, "step_ms": elapsed * 1000.0 / steps, "hz": steps / elapsed}


def run_template_search(
    *,
    workspace: Path,
    artifact_dir: Path,
    profile: str = "quick",
    data_root: Path | str | None = None,
) -> dict:
    dataset = load_dataset(data_root if data_root is not None else default_data_root(workspace))
    train, val, test = split_trials(dataset.trials)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir = artifact_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if profile == "quick":
        feature_cfgs = [
            FeatureConfig(channels=ch, smooth_alpha=alpha, history=hist, lag=0)
            for ch in ["all", "motor"]
            for alpha in [0.15]
            for hist in [1, 3]
        ]
        ridge_alphas = [1.0, 10.0, 100.0, 1000.0]
        running_options = [False, True]
        prob_alphas = [0.05, 0.15, 0.35, 0.70, 1.0]
        temps = [0.10, 0.25, 0.50, 1.0]
        output_alphas = [0.5, 0.75, 1.0]
    else:
        feature_cfgs = [
            FeatureConfig(channels=ch, smooth_alpha=alpha, history=hist, lag=0)
            for ch in ["all", "motor", "dM1", "vM1", "no_pMC"]
            for alpha in [None, 0.05, 0.15, 0.30]
            for hist in [1, 2, 3, 5]
        ]
        ridge_alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
        running_options = [False, True]
        prob_alphas = [0.03, 0.08, 0.15, 0.35, 0.70, 1.0]
        temps = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
        output_alphas = [0.35, 0.5, 0.75, 1.0]

    rows = []
    best_policy = None
    best_score = -np.inf
    idx = 0
    t_all = time.perf_counter()
    for fc in feature_cfgs:
        for ridge in ridge_alphas:
            for use_running_mean in running_options:
                base = fit_template_policy(dataset, train, fc, ridge, use_running_mean=use_running_mean)
                for pa in prob_alphas:
                    for temp in temps:
                        for oa in output_alphas:
                            idx += 1
                            base.prob_alpha = pa
                            base.temperature = temp
                            base.output_alpha = oa
                            yv, pv, gv = simulate_template(base, val)
                            m = summarize(yv, pv)
                            true_g = np.concatenate(
                                [np.full(t.n_time - fc.lag, t.gesture, dtype=np.int64) for t in val]
                            )
                            acc = float(np.mean(gv == true_g))
                            feature_dim = int(base.class_weights.shape[0])
                            row = {
                                **m,
                                "gesture_acc": acc,
                                "idx": idx,
                                "channels": fc.channels,
                                "smooth_alpha": np.nan if fc.smooth_alpha is None else fc.smooth_alpha,
                                "history": fc.history,
                                "ridge_alpha": ridge,
                                "prob_alpha": pa,
                                "temperature": temp,
                                "output_alpha": oa,
                                "use_running_mean": use_running_mean,
                                "n_channels": int(base.channel_indices.size),
                                "n_features": feature_dim,
                            }
                            rows.append(row)
                            score = m["R2"] + 0.03 * acc - 0.01 * m["jitter"]
                            if score > best_score:
                                best_score = score
                                best_policy = TemplatePolicy(
                                    feature_config=fc,
                                    channel_indices=base.channel_indices.copy(),
                                    x_scaler=base.x_scaler,
                                    class_weights=base.class_weights.copy(),
                                    class_bias=base.class_bias.copy(),
                                    templates_qpos=[x.copy() for x in base.templates_qpos],
                                    templates_sensor=[x.copy() for x in base.templates_sensor],
                                    prob_alpha=pa,
                                    temperature=temp,
                                    output_alpha=oa,
                                    use_running_mean=use_running_mean,
                                )
                if idx % 240 == 0:
                    print(
                        f"[template {idx}] best_score={best_score:+.4f} "
                        f"last_R2={rows[-1]['R2']:+.4f} elapsed={(time.perf_counter()-t_all)/60:.1f}m",
                        flush=True,
                    )
                    pd.DataFrame(rows).sort_values("R2", ascending=False).to_csv(
                        results_dir / "template_leaderboard_partial.csv", index=False
                    )

    if best_policy is None:
        raise RuntimeError("No template policy trained.")

    leaderboard = pd.DataFrame(rows)
    leaderboard.sort_values(["R2", "gesture_acc"], ascending=False).to_csv(
        results_dir / "template_leaderboard.csv", index=False
    )
    yt, pt, gt = simulate_template(best_policy, test)
    test_metrics = summarize(yt, pt)
    true_gt = np.concatenate(
        [np.full(t.n_time - best_policy.feature_config.lag, t.gesture, dtype=np.int64) for t in test]
    )
    lat = template_latency(best_policy, test[0])
    best_policy.save(artifact_dir / "best_template_policy.npz")
    stable = workspace / "artifacts"
    stable.mkdir(parents=True, exist_ok=True)
    best_policy.save(stable / "best_template_policy.npz")
    summary = {
        "profile": profile,
        "n_candidates": idx,
        "best_validation": leaderboard.sort_values(["R2", "gesture_acc"], ascending=False).iloc[0].to_dict(),
        "test_online": test_metrics,
        "test_gesture_acc": float(np.mean(gt == true_gt)),
        "latency": lat,
        "policy_file": str(artifact_dir / "best_template_policy.npz"),
    }
    (artifact_dir / "template_summary.json").write_text(json.dumps(summary, indent=2))
    (stable / "template_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
