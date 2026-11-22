#!/usr/bin/env python3
"""Advanced policy helpers used by recurrent and Kalman-style experiments.

The current split/PCA suite imports selected functions from this file rather
than duplicating earlier GRU/LFADS/Kalman utilities.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder_core.data import (
    FeatureConfig,
    Trial,
    causal_ema,
    default_data_root,
    load_dataset,
    split_trials,
)
from decoder_core.metrics import summarize
from decoder_core.models import Standardizer, _fit_ridge_readout
from decoder_core.template_policy import build_template_features

from run_extra_experiments import AffineKalman, fit_affine_kalman, run_affine_kalman


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


def target_for_trial(trial: Trial, target_mode: str) -> np.ndarray:
    if target_mode == "qpos":
        return trial.qpos
    if target_mode == "qvel":
        return trial.qvel
    if target_mode == "pv":
        return np.concatenate([trial.qpos, trial.qvel], axis=1).astype(np.float32)
    if target_mode == "ctrl":
        return trial.ctrl
    raise ValueError(target_mode)


def concat_targets(trials: list[Trial], target_mode: str) -> np.ndarray:
    return np.concatenate([target_for_trial(t, target_mode) for t in trials], axis=0).astype(np.float32)


def trial_phase(trial: Trial) -> np.ndarray:
    return np.linspace(0.0, 1.0, trial.n_time, dtype=np.float32)


def phase_velocity(trial: Trial) -> np.ndarray:
    return np.full(trial.n_time, 1.0 / max(1, trial.n_time - 1), dtype=np.float32)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    z = scores / temp
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z).astype(np.float32)
    return e / np.sum(e, axis=1, keepdims=True)


def interp_template(template: np.ndarray, phase: np.ndarray) -> np.ndarray:
    phase = np.clip(np.asarray(phase, dtype=np.float32), 0.0, 1.0)
    x = phase * np.float32(template.shape[0] - 1)
    lo = np.floor(x).astype(np.int64)
    hi = np.minimum(lo + 1, template.shape[0] - 1)
    w = (x - lo.astype(np.float32))[:, None]
    return ((1.0 - w) * template[lo] + w * template[hi]).astype(np.float32)


def smooth_probabilities(probs: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(probs, dtype=np.float32)
    out[0] = probs[0]
    a = np.float32(alpha)
    for i in range(1, probs.shape[0]):
        out[i] = (np.float32(1.0) - a) * out[i - 1] + a * probs[i]
        out[i] /= np.sum(out[i])
    return out


def smooth_outputs(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 0.999:
        return y.astype(np.float32, copy=True)
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    a = np.float32(alpha)
    for i in range(1, y.shape[0]):
        out[i] = a * y[i] + (np.float32(1.0) - a) * out[i - 1]
    return out


class KalmanTemplatePolicy:
    def __init__(
        self,
        *,
        target_mode: str,
        cfg: FeatureConfig,
        channels: np.ndarray,
        obs_scaler: Standardizer,
        kalman: AffineKalman,
        class_weights: np.ndarray,
        class_bias: np.ndarray,
        templates: list[np.ndarray],
        temperature: float,
        prob_alpha: float,
        output_alpha: float,
    ):
        self.target_mode = target_mode
        self.cfg = cfg
        self.channels = channels
        self.obs_scaler = obs_scaler
        self.kalman = kalman
        self.class_weights = class_weights
        self.class_bias = class_bias
        self.templates = templates
        self.temperature = float(temperature)
        self.prob_alpha = float(prob_alpha)
        self.output_alpha = float(output_alpha)

    def _obs(self, trial: Trial) -> np.ndarray:
        raw = build_template_features(trial, self.channels, self.cfg, use_running_mean=True)
        return self.obs_scaler.transform(raw).astype(np.float32)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        obs = self._obs(trial)
        state = run_affine_kalman(self.kalman, obs)
        phase = np.maximum.accumulate(np.clip(state[:, 0], 0.0, 1.0))
        logits = obs @ self.class_weights + self.class_bias
        return reconstruct_from_phase_logits(
            self.templates,
            phase,
            logits,
            temperature=self.temperature,
            prob_alpha=self.prob_alpha,
            output_alpha=self.output_alpha,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "metadata": np.asarray(
                json.dumps(
                    {
                        "model_family": "kalman_phase_template",
                        "target_mode": self.target_mode,
                        "feature_config": asdict(self.cfg),
                        "temperature": self.temperature,
                        "prob_alpha": self.prob_alpha,
                        "output_alpha": self.output_alpha,
                    }
                )
            ),
            "channels": self.channels.astype(np.int64),
            "obs_mean": self.obs_scaler.mean,
            "obs_scale": self.obs_scaler.scale,
            "class_weights": self.class_weights,
            "class_bias": self.class_bias,
            "kalman_a": self.kalman.a,
            "kalman_b": self.kalman.b,
            "kalman_h": self.kalman.h,
            "kalman_d": self.kalman.d,
            "kalman_w": self.kalman.w,
            "kalman_r_diag": self.kalman.r_diag,
            "kalman_x0": self.kalman.x0,
            "kalman_p0": self.kalman.p0,
            "kalman_htrh": self.kalman.htrh,
        }
        for i, template in enumerate(self.templates):
            arrays[f"template_{i}"] = template.astype(np.float32)
        np.savez_compressed(path, **arrays)


def reconstruct_from_phase_logits(
    templates: list[np.ndarray],
    phase: np.ndarray,
    logits: np.ndarray,
    *,
    temperature: float,
    prob_alpha: float,
    output_alpha: float,
) -> np.ndarray:
    probs = smooth_probabilities(softmax(logits, temperature), prob_alpha)
    y = np.zeros((phase.shape[0], templates[0].shape[1]), dtype=np.float32)
    for gesture, template in enumerate(templates):
        y += probs[:, gesture : gesture + 1] * interp_template(template, phase)
    return smooth_outputs(y, output_alpha)


def fit_base_kalman(train: list[Trial]) -> tuple[FeatureConfig, np.ndarray, Standardizer, AffineKalman, np.ndarray, np.ndarray]:
    cfg = FeatureConfig(channels="all", smooth_alpha=0.15, history=1, lag=0)
    channels = np.arange(256, dtype=np.int64)
    raw_obs = [build_template_features(t, channels, cfg, use_running_mean=True) for t in train]
    obs_scaler = Standardizer.fit(np.concatenate(raw_obs, axis=0).astype(np.float32))
    obs_trials = [obs_scaler.transform(x).astype(np.float32) for x in raw_obs]

    state_trials = [
        np.stack([trial_phase(t), phase_velocity(t)], axis=1).astype(np.float32)
        for t in train
    ]
    x0 = np.asarray([0.0, np.median(np.concatenate([s[:, 1] for s in state_trials]))], dtype=np.float32)
    kalman = fit_affine_kalman(state_trials, obs_trials, x0=x0)

    labels = np.concatenate([np.full(t.n_time, t.gesture, dtype=np.int64) for t in train])
    y = np.zeros((labels.size, max(labels) + 1), dtype=np.float32)
    y[np.arange(labels.size), labels] = 1.0
    class_weights, class_bias = _fit_ridge_readout(
        obs_scaler.transform(np.concatenate(raw_obs, axis=0).astype(np.float32)).astype(np.float32),
        y,
        alpha=1000.0,
    )
    return cfg, channels, obs_scaler, kalman, class_weights, class_bias


def fit_kalman_template_grid(
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    target_mode: str,
    out_dir: Path,
) -> tuple[dict, KalmanTemplatePolicy, list[np.ndarray], list[np.ndarray]]:
    cfg, channels, obs_scaler, kalman, class_weights, class_bias = fit_base_kalman(train)
    templates = []
    for gesture in range(max(t.gesture for t in train) + 1):
        candidates = [t for t in train if t.gesture == gesture]
        templates.append(target_for_trial(candidates[0], target_mode).copy())

    def prepare_latents(trials: list[Trial]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        prepared = []
        for trial in trials:
            raw = build_template_features(trial, channels, cfg, use_running_mean=True)
            obs = obs_scaler.transform(raw).astype(np.float32)
            state = run_affine_kalman(kalman, obs)
            phase = np.maximum.accumulate(np.clip(state[:, 0], 0.0, 1.0))
            logits = obs @ class_weights + class_bias
            prepared.append((target_for_trial(trial, target_mode), phase, logits))
        return prepared

    val_prepared = prepare_latents(val)
    test_prepared = prepare_latents(test)
    val_true_concat = np.concatenate([p[0] for p in val_prepared], axis=0)

    best_policy: KalmanTemplatePolicy | None = None
    best_row: dict | None = None
    rows = []
    for temperature in [0.1, 0.25, 0.5, 1.0]:
        for prob_alpha in [0.05, 0.15, 0.35, 0.7]:
            for output_alpha in [0.55, 0.75, 0.9, 1.0]:
                policy = KalmanTemplatePolicy(
                    target_mode=target_mode,
                    cfg=cfg,
                    channels=channels,
                    obs_scaler=obs_scaler,
                    kalman=kalman,
                    class_weights=class_weights,
                    class_bias=class_bias,
                    templates=templates,
                    temperature=temperature,
                    prob_alpha=prob_alpha,
                    output_alpha=output_alpha,
                )
                val_pred = np.concatenate(
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
                val_metrics = summarize(val_true_concat, val_pred)
                row = {
                    "model_family": "kalman_phase_template",
                    "target_mode": target_mode,
                    "temperature": temperature,
                    "prob_alpha": prob_alpha,
                    "output_alpha": output_alpha,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                rows.append(row)
                score = val_metrics["R2"] + 0.03 * val_metrics["CC"] - 0.01 * val_metrics["jitter"]
                if best_row is None or score > best_row["_score"]:
                    best_row = {**row, "_score": score}
                    best_policy = policy

    assert best_policy is not None and best_row is not None
    test_true = np.concatenate([p[0] for p in test_prepared], axis=0)
    test_preds_by_trial = [
        reconstruct_from_phase_logits(
            templates,
            phase,
            logits,
            temperature=best_policy.temperature,
            prob_alpha=best_policy.prob_alpha,
            output_alpha=best_policy.output_alpha,
        )
        for _, phase, logits in test_prepared
    ]
    test_pred = np.concatenate(test_preds_by_trial, axis=0)
    test_metrics = summarize(test_true, test_pred)
    row = {k: v for k, v in best_row.items() if k != "_score"}
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    policy_path = out_dir / f"best_kalman_phase_template_{target_mode}.npz"
    best_policy.save(policy_path)
    row["artifact"] = str(policy_path)
    pd.DataFrame(rows).sort_values(["val_R2", "val_CC"], ascending=False).to_csv(
        out_dir / f"kalman_phase_template_{target_mode}_grid.csv",
        index=False,
    )
    return row, best_policy, [target_for_trial(t, target_mode) for t in test], test_preds_by_trial


def predict_trials(predict_one: Callable[[Trial], np.ndarray], trials: list[Trial], target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    trues, preds = [], []
    for trial in trials:
        pred = predict_one(trial)
        true = target_for_trial(trial, target_mode)
        n = min(true.shape[0], pred.shape[0])
        trues.append(true[:n])
        preds.append(pred[:n])
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def torch_available():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for GRU and LFADS-lite training. Install torch and rerun.") from exc
    return torch, nn


def make_sequence_arrays(
    trials: list[Trial],
    target_mode: str,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    smooth_alpha: float = 0.15,
) -> list[tuple[np.ndarray, np.ndarray]]:
    seqs = []
    for trial in trials:
        x = causal_ema(trial.neural, smooth_alpha)
        y = target_for_trial(trial, target_mode)
        n = min(x.shape[0], y.shape[0])
        seqs.append(
            (
                x_scaler.transform(x[:n]).astype(np.float32),
                y_scaler.transform(y[:n]).astype(np.float32),
            )
        )
    return seqs


def pad_batch(batch, torch, device):
    lengths = np.asarray([x.shape[0] for x, _ in batch], dtype=np.int64)
    max_len = int(lengths.max())
    input_dim = batch[0][0].shape[1]
    output_dim = batch[0][1].shape[1]
    xb = np.zeros((len(batch), max_len, input_dim), dtype=np.float32)
    yb = np.zeros((len(batch), max_len, output_dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=bool)
    for i, (x, y) in enumerate(batch):
        n = x.shape[0]
        xb[i, :n] = x
        yb[i, :n] = y
        mask[i, :n] = True
    return (
        torch.from_numpy(xb).to(device),
        torch.from_numpy(yb).to(device),
        torch.from_numpy(mask).to(device),
    )


def masked_mse(pred, target, mask):
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target) ** 2 * mask_f).sum() / (mask_f.sum() * pred.shape[-1]).clamp_min(1.0)


def masked_delta_mse(pred, target, mask):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(pred.dtype)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return ((dp - dt) ** 2 * valid).sum() / (valid.sum() * pred.shape[-1]).clamp_min(1.0)


def iter_batches(seqs, batch_size: int, rng: np.random.Generator, shuffle: bool):
    indices = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [seqs[i] for i in indices[start : start + batch_size]]


def train_torch_model(
    *,
    model_kind: str,
    target_mode: str,
    train: list[Trial],
    val: list[Trial],
    test: list[Trial],
    out_dir: Path,
    hidden_dim: int,
    factor_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    recon_weight: float,
    smooth_weight: float,
    device_name: str,
) -> tuple[dict, list[np.ndarray]]:
    torch, nn = torch_available()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)

    class GRUDecoder(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(self.norm(h)), {}

    class LFADSLiteDecoder(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.factor = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, factor_dim), nn.Tanh())
            self.target_head = nn.Sequential(nn.Linear(factor_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
            self.recon_head = nn.Linear(factor_dim, input_dim)

        def forward(self, x):
            h, _ = self.gru(x)
            factors = self.factor(h)
            return self.target_head(factors), {"recon": self.recon_head(factors), "factors": factors}

    x_scaler = Standardizer.fit(np.concatenate([causal_ema(t.neural, 0.15) for t in train], axis=0).astype(np.float32))
    y_scaler = Standardizer.fit(concat_targets(train, target_mode))
    train_seqs = make_sequence_arrays(train, target_mode, x_scaler, y_scaler)
    val_seqs = make_sequence_arrays(val, target_mode, x_scaler, y_scaler)
    test_seqs = make_sequence_arrays(test, target_mode, x_scaler, y_scaler)
    input_dim = train_seqs[0][0].shape[1]
    output_dim = train_seqs[0][1].shape[1]
    if model_kind == "gru":
        model = GRUDecoder(input_dim, output_dim)
    elif model_kind == "lfads_lite":
        model = LFADSLiteDecoder(input_dim, output_dim)
    else:
        raise ValueError(model_kind)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in iter_batches(train_seqs, batch_size, rng, shuffle=True):
            xb, yb, mask = pad_batch(batch, torch, device)
            pred, extra = model(xb)
            loss = masked_mse(pred, yb, mask) + smooth_weight * masked_delta_mse(pred, yb, mask)
            if model_kind == "lfads_lite":
                loss = loss + recon_weight * masked_mse(extra["recon"], xb, mask)
                loss = loss + 1e-4 * (extra["factors"] ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss = evaluate_loss(model, val_seqs, torch, device, batch_size)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    train_ms = (time.perf_counter() - t0) * 1000.0
    val_true, val_pred = predict_torch(model, val, val_seqs, y_scaler, torch, device, batch_size, target_mode)
    test_true, test_pred = predict_torch(model, test, test_seqs, y_scaler, torch, device, batch_size, target_mode)
    val_metrics = summarize(val_true, val_pred)
    test_metrics = summarize(test_true, test_pred)
    test_by_trial = predict_torch_by_trial(model, test_seqs, y_scaler, torch, device)
    artifact = out_dir / f"best_{model_kind}_{target_mode}.pt"
    torch.save(
        {
            "model_kind": model_kind,
            "target_mode": target_mode,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "factor_dim": factor_dim,
            "state_dict": best_state,
            "x_mean": x_scaler.mean,
            "x_scale": x_scaler.scale,
            "y_mean": y_scaler.mean,
            "y_scale": y_scaler.scale,
            "smooth_alpha": 0.15,
            "recon_weight": recon_weight,
            "smooth_weight": smooth_weight,
        },
        artifact,
    )
    row = {
        "model_family": model_kind,
        "target_mode": target_mode,
        "hidden_dim": hidden_dim,
        "factor_dim": factor_dim if model_kind == "lfads_lite" else np.nan,
        "lr": lr,
        "epochs_ran": best_epoch,
        "train_ms": train_ms,
        "recon_weight": recon_weight if model_kind == "lfads_lite" else np.nan,
        "smooth_weight": smooth_weight,
        "artifact": str(artifact),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return row, test_by_trial


def evaluate_loss(model, seqs, torch, device, batch_size: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in iter_batches(seqs, batch_size, np.random.default_rng(0), shuffle=False):
            xb, yb, mask = pad_batch(batch, torch, device)
            pred, _ = model(xb)
            losses.append(float(masked_mse(pred, yb, mask).detach().cpu()))
    return float(np.mean(losses))


def predict_torch_by_trial(model, seqs, y_scaler: Standardizer, torch, device) -> list[np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in seqs:
            xb = torch.from_numpy(x[None, :, :]).to(device)
            pred, _ = model(xb)
            y = pred[0].detach().cpu().numpy().astype(np.float32)
            preds.append(y_scaler.inverse(y).astype(np.float32))
    return preds


def predict_torch(model, trials, seqs, y_scaler: Standardizer, torch, device, batch_size: int, target_mode: str):
    preds = predict_torch_by_trial(model, seqs, y_scaler, torch, device)
    trues = [target_for_trial(t, target_mode)[: p.shape[0]] for t, p in zip(trials, preds, strict=True)]
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def export_compare_predictions(
    *,
    test: list[Trial],
    predictions: list[np.ndarray],
    out_dir: Path,
    policy_name: str,
    repeat: int,
    source_artifact: str,
    metrics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for trial, pred in zip(test, predictions, strict=True):
        if trial.repeat != repeat:
            continue
        n = min(trial.ctrl.shape[0], pred.shape[0])
        actual_path = out_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_actual_ctrl.npy"
        decoded_path = out_dir / f"gesture_{trial.gesture:02d}_repeat_{trial.repeat:02d}_decoded_ctrl.npy"
        np.save(actual_path, trial.ctrl[:n].T.astype(np.float32))
        np.save(decoded_path, pred[:n].T.astype(np.float32))
        exports.append(
            {
                "gesture": trial.gesture,
                "repeat": trial.repeat,
                "frames": int(n),
                "actual": str(actual_path),
                "decoded": str(decoded_path),
                "decoded_note": f"Decoded ctrl from {policy_name}.",
            }
        )
    summary = {
        "policy": policy_name,
        "source_artifact": source_artifact,
        "repeat_requested": repeat,
        "order_note": "Both arrays are in MuJoCo ctrl order and have shape 32 x time.",
        "metrics": clean_dict(metrics),
        "exports": exports,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def profile_settings(profile: str) -> dict:
    if profile == "full":
        return {"epochs": 120, "patience": 18, "batch_size": 10, "grids": [(96, 1e-3), (128, 8e-4)], "lfads_recon": [0.03, 0.1]}
    if profile == "quick":
        return {"epochs": 45, "patience": 10, "batch_size": 12, "grids": [(96, 1e-3)], "lfads_recon": [0.05]}
    raise ValueError(profile)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Kalman, GRU, and LFADS-style decoders on the attempted movement data.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/advanced_policies"))
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-repeat", type=int, default=8)
    args = parser.parse_args()

    data_root = args.data_root if args.data_root is not None else default_data_root(args.workspace)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_root)
    train, val, test = split_trials(dataset.trials)
    settings = profile_settings(args.profile)
    rows = []
    best_ctrl = {"val_R2": -np.inf, "predictions": None, "row": None}

    print(f"Loaded {len(dataset.trials)} trials: train={len(train)} val={len(val)} test={len(test)}", flush=True)
    print("Training Kalman phase/template policies...", flush=True)
    for target_mode in ["qpos", "ctrl"]:
        row, _, _, preds = fit_kalman_template_grid(train, val, test, target_mode, args.out_dir)
        rows.append(row)
        if target_mode == "ctrl" and row["val_R2"] > best_ctrl["val_R2"]:
            best_ctrl = {"val_R2": row["val_R2"], "predictions": preds, "row": row}
        print(
            f"  Kalman {target_mode}: val_R2={row['val_R2']:.3f} test_R2={row['test_R2']:.3f} test_CC={row['test_CC']:.3f}",
            flush=True,
        )

    print("Training GRU and LFADS-lite sequence policies...", flush=True)
    for target_mode in ["qpos", "ctrl"]:
        for hidden_dim, lr in settings["grids"]:
            row, preds = train_torch_model(
                model_kind="gru",
                target_mode=target_mode,
                train=train,
                val=val,
                test=test,
                out_dir=args.out_dir,
                hidden_dim=hidden_dim,
                factor_dim=32,
                lr=lr,
                epochs=settings["epochs"],
                patience=settings["patience"],
                batch_size=settings["batch_size"],
                seed=args.seed,
                recon_weight=0.0,
                smooth_weight=0.03,
                device_name=args.device,
            )
            rows.append(row)
            if target_mode == "ctrl" and row["val_R2"] > best_ctrl["val_R2"]:
                best_ctrl = {"val_R2": row["val_R2"], "predictions": preds, "row": row}
            print(
                f"  GRU {target_mode}: val_R2={row['val_R2']:.3f} test_R2={row['test_R2']:.3f} test_CC={row['test_CC']:.3f}",
                flush=True,
            )

            for recon_weight in settings["lfads_recon"]:
                row, preds = train_torch_model(
                    model_kind="lfads_lite",
                    target_mode=target_mode,
                    train=train,
                    val=val,
                    test=test,
                    out_dir=args.out_dir,
                    hidden_dim=hidden_dim,
                    factor_dim=32,
                    lr=lr,
                    epochs=settings["epochs"],
                    patience=settings["patience"],
                    batch_size=settings["batch_size"],
                    seed=args.seed + 1,
                    recon_weight=recon_weight,
                    smooth_weight=0.03,
                    device_name=args.device,
                )
                rows.append(row)
                if target_mode == "ctrl" and row["val_R2"] > best_ctrl["val_R2"]:
                    best_ctrl = {"val_R2": row["val_R2"], "predictions": preds, "row": row}
                print(
                    f"  LFADS-lite {target_mode}: val_R2={row['val_R2']:.3f} test_R2={row['test_R2']:.3f} test_CC={row['test_CC']:.3f}",
                    flush=True,
                )

    table = pd.DataFrame([clean_dict(row) for row in rows]).sort_values(["target_mode", "val_R2"], ascending=[True, False])
    leaderboard_path = args.out_dir / "leaderboard.csv"
    table.to_csv(leaderboard_path, index=False)

    if best_ctrl["predictions"] is not None and best_ctrl["row"] is not None:
        compare_dir = args.out_dir / "mujoco_compare_best_ctrl"
        export_compare_predictions(
            test=test,
            predictions=best_ctrl["predictions"],
            out_dir=compare_dir,
            policy_name=f"{best_ctrl['row']['model_family']}_{best_ctrl['row']['target_mode']}",
            repeat=args.export_repeat,
            source_artifact=str(best_ctrl["row"]["artifact"]),
            metrics=best_ctrl["row"],
        )
    else:
        compare_dir = None

    summary = {
        "profile": args.profile,
        "data_root": str(data_root),
        "split": {"train_repeats": [0, 1, 2, 3, 4, 5], "val_repeats": [6, 7], "test_repeats": [8, 9]},
        "leaderboard": str(leaderboard_path),
        "best_by_target": {
            target: clean_dict(group.sort_values(["val_R2", "val_CC"], ascending=False).iloc[0].to_dict())
            for target, group in table.groupby("target_mode")
        },
        "best_ctrl_compare_dir": str(compare_dir) if compare_dir is not None else None,
        "notes": {
            "kalman": "Causal affine Kalman filter estimates elapsed phase from neural activity; a smoothed intent classifier chooses/blends gesture templates.",
            "gru": "Causal GRU maps 256-channel neural sequences directly to qpos or ctrl.",
            "lfads_lite": "LFADS-inspired supervised latent-factor GRU with neural reconstruction regularization; not a full unsupervised LFADS publication implementation.",
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote leaderboard: {leaderboard_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    if compare_dir is not None:
        print(f"Wrote MuJoCo comparison exports: {compare_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
