#!/usr/bin/env python3
"""BRAND-style gated streaming decoder scaffold.

This file is intentionally dependency-light so it can be tested before the
final BRAND API names are known.  The core class, StreamingQposDecoder, does the
part that matters for BRAND:

1. receive one 100 Hz neural bin;
2. check a decode/rest flag;
3. if decoding is active, build the same features used in training;
4. run a saved GRU/LFADS-lite/Transformer artifact;
5. convert the artifact output into a 32-D qpos command when needed;
6. publish one 32-D qpos command with a timestamp.

The bottom CLI simulates the streams with artificial neural data.  In the lab,
the small placeholder methods read_brand_inputs() and publish_brand_output()
should be connected to the actual BRAND subscribe/publish APIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import time
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from scripts.run_split_pca_model_suite import SequenceNetFactory, N_QPOS


NODE_STARTED = "NODE_STARTED"
NODE_READY = "NODE_READY"
NODE_SHUTDOWN = "NODE_SHUTDOWN"
NODE_FATAL_ERROR = "NODE_FATAL_ERROR"

# New VM sessions use six MuJoCo object/body IDs.  The strict same-threshold
# new-only artifacts normally use compact object columns 0..5.  Some older
# pooled old+new artifacts kept old objects in columns 0..5 and appended the VM
# objects in columns 6..11.  The helper below chooses the offset from the
# artifact's object-label width, so both formats can be tested.
VM_BODY_ID_TO_COMPACT_OBJECT_ID = {
    55: 0,  # small_ball
    56: 1,  # big_ball
    57: 2,  # small_v_cylinder
    58: 3,  # big_v_cylinder
    59: 4,  # small_h_cylinder
    60: 5,  # big_h_cylinder
}

VM_COMPACT_OBJECT_ID_TO_NAME = {
    0: "small_ball",
    1: "big_ball",
    2: "small_v_cylinder",
    3: "big_v_cylinder",
    4: "small_h_cylinder",
    5: "big_h_cylinder",
}


def install_numpy_pickle_compat() -> None:
    """Let older NumPy environments load artifacts saved by newer NumPy."""
    # Some VM artifacts were saved with a NumPy version that pickles arrays
    # under numpy._core.  Older local environments expose the same internals as
    # numpy.core.  These aliases affect unpickling only; they do not modify data.
    try:
        import numpy.core as np_core

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        pass


@dataclass(slots=True)
class StreamStep:
    """One output packet that a BRAND publisher would send downstream."""

    timestamp_s: float
    qpos: np.ndarray
    phase: float
    latency_ms: float
    object_id: int | None
    raw_object_id: int | None = None


class StreamingQposDecoder:
    """Online-style wrapper around one saved causal sequence decoder artifact."""

    def __init__(
        self,
        artifact_path: Path,
        *,
        device: str = "cpu",
        sample_rate_hz: float = 100.0,
        expected_duration_s: float = 4.5,
        neural_smooth_alpha: float = 0.15,
        neural_already_smoothed: bool = False,
        max_context_bins: int = 240,
        default_object_id: int | None = None,
        template_prior: np.ndarray | None = None,
        initial_qpos: np.ndarray | None = None,
    ) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device)
        self.sample_rate_hz = float(sample_rate_hz)
        self.expected_duration_s = float(expected_duration_s)
        self.neural_smooth_alpha = float(neural_smooth_alpha)
        self.neural_already_smoothed = bool(neural_already_smoothed)
        self.max_context_bins = int(max_context_bins)
        self.default_object_id = default_object_id
        self.template_prior = template_prior
        self.initial_qpos = (
            np.zeros(N_QPOS, dtype=np.float32)
            if initial_qpos is None
            else np.asarray(initial_qpos, dtype=np.float32).reshape(N_QPOS)
        )
        self.current_qpos = self.initial_qpos.copy()

        install_numpy_pickle_compat()
        payload = torch.load(Path(artifact_path), map_location=self.device, weights_only=False)
        self.model_family = str(payload["model_family"])
        self.model_mode = str(payload["model_mode"])
        self.condition_object = bool(payload.get("condition_object", False))
        self.projector = payload.get("projector", {"mode": "none", "mean": None, "components": None})

        self.x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
        self.x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
        self.y_mean = np.asarray(payload["y_mean"], dtype=np.float32)
        self.y_scale = np.asarray(payload["y_scale"], dtype=np.float32)
        self.target_mode = str(payload.get("target_mode", "qpos"))
        self.target_base_mode = str(payload.get("target_base_mode", self._base_mode_from_target(self.target_mode)))
        self.target_mean = payload.get("target_mean", None)
        self.target_components = payload.get("target_components", None)
        if self.target_mean is not None:
            self.target_mean = np.asarray(self.target_mean, dtype=np.float32)
        if self.target_components is not None:
            self.target_components = np.asarray(self.target_components, dtype=np.float32)

        input_dim = int(self.x_mean.shape[0])
        output_dim = int(payload.get("output_dim", self.y_mean.shape[0]))
        hidden_dim = int(payload["hidden_dim"])
        factor_dim = int(payload.get("factor_dim", 32))
        gru_layers = int(payload.get("gru_layers", 1))
        self.model = SequenceNetFactory.make(
            torch,
            nn,
            model_family=self.model_family,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            factor_dim=factor_dim,
            bidirectional_lfads=False,
            gru_layers=gru_layers,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.neural_dim = self._neural_feature_dim()
        self.n_objects = input_dim - self.neural_dim - 1
        if self.model_mode == "hybrid":
            self.n_objects -= N_QPOS
            if self.template_prior is None:
                raise ValueError(
                    "Hybrid artifacts need a template_prior for online use. "
                    "For the first BRAND node, prefer repeat_holdout/no-PCA/direct GRU or LFADS."
                )
        if self.n_objects < 0:
            raise ValueError(f"Could not infer object-label width from artifact input_dim={input_dim}.")

        self.context: deque[np.ndarray] = deque(maxlen=self.max_context_bins)
        self.rnn_hidden = None
        self.ema_state: np.ndarray | None = None
        self.decode_active = False
        self.bin_in_attempt = 0

    def _neural_feature_dim(self) -> int:
        """Return neural feature size after optional PCA."""
        mode = str(self.projector.get("mode", "none"))
        components = self.projector.get("components")
        if mode == "none" or components is None:
            return 256
        return int(np.asarray(components).shape[0])

    def reset_attempt(self) -> None:
        """Clear causal state at the start or end of an attempted movement."""
        self.context.clear()
        self.rnn_hidden = None
        self.ema_state = None
        self.decode_active = False
        self.bin_in_attempt = 0
        self.current_qpos = self.initial_qpos.copy()

    @staticmethod
    def _base_mode_from_target(target_mode: str) -> str:
        """Infer whether an artifact predicts qpos, qvel, or qpos+qvel."""
        mode = str(target_mode)
        if mode.startswith("qvel"):
            return "qvel"
        if mode.startswith("pv") or mode.startswith("qpos_qvel"):
            return "pv"
        return "qpos"

    def _causal_neural_features(self, neural_bin_256: np.ndarray) -> np.ndarray:
        """Apply the same causal smoothing and optional PCA used during training."""
        x = np.asarray(neural_bin_256, dtype=np.float32).reshape(-1)
        if x.shape[0] != 256:
            raise ValueError(f"Expected one 256-channel neural bin, got shape {x.shape}.")
        if not self.neural_already_smoothed:
            if self.ema_state is None:
                self.ema_state = x.copy()
            else:
                alpha = np.float32(self.neural_smooth_alpha)
                self.ema_state = alpha * x + (np.float32(1.0) - alpha) * self.ema_state
            x = self.ema_state
        if str(self.projector.get("mode", "none")) != "none":
            mean = np.asarray(self.projector["mean"], dtype=np.float32)
            components = np.asarray(self.projector["components"], dtype=np.float32)
            x = (x - mean) @ components.T
        return x.astype(np.float32)

    def _phase(self) -> float:
        """Elapsed attempt phase in [0, 1], generated from the 100 Hz decode clock."""
        denom = max(1.0, self.expected_duration_s * self.sample_rate_hz - 1.0)
        return float(np.clip(self.bin_in_attempt / denom, 0.0, 1.0))

    def _template_at_phase(self, phase: float) -> np.ndarray:
        """Interpolate a 32-D template prior at the current phase for hybrid mode."""
        assert self.template_prior is not None
        template = np.asarray(self.template_prior, dtype=np.float32)
        if template.shape[0] == N_QPOS:
            template = template.T
        x = phase * (template.shape[0] - 1)
        lo = int(np.floor(x))
        hi = min(lo + 1, template.shape[0] - 1)
        w = np.float32(x - lo)
        return ((1.0 - w) * template[lo] + w * template[hi]).astype(np.float32)

    def _decode_target_space(self, y: np.ndarray) -> np.ndarray:
        """Convert model target output back to physical qpos/qvel/pv values."""
        y = np.asarray(y, dtype=np.float32)
        if self.target_mode == "qpos" or self.target_components is None:
            return y
        if self.target_mean is None:
            raise ValueError("Artifact has target PCA components but no target mean.")
        return (y @ self.target_components + self.target_mean).astype(np.float32)

    def _target_to_qpos(self, target: np.ndarray) -> np.ndarray:
        """Convert qpos/qvel/qpos+qvel model output into a 32-D qpos command."""
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if self.target_base_mode == "qpos":
            self.current_qpos = target[:N_QPOS].astype(np.float32)
            return self.current_qpos.copy()
        if self.target_base_mode == "pv":
            self.current_qpos = target[:N_QPOS].astype(np.float32)
            return self.current_qpos.copy()
        if self.target_base_mode == "qvel":
            dt = np.float32(1.0 / self.sample_rate_hz)
            self.current_qpos = (self.current_qpos + target[:N_QPOS] * dt).astype(np.float32)
            return self.current_qpos.copy()
        raise ValueError(f"Unsupported target_base_mode: {self.target_base_mode}")

    def _object_id_for_model(self, object_id: int | None) -> int | None:
        """Convert online object IDs into the compact class ID expected by the model."""
        raw = self.default_object_id if object_id is None else object_id
        if raw is None:
            return None
        raw = int(raw)
        if self.n_objects == 0:
            return None

        # Strict new-only artifacts normally have six object columns, so no
        # offset is used.  Older pooled artifacts may have twelve columns; then
        # VM objects are shifted to columns 6..11.
        vm_offset = 6 if self.n_objects >= 12 else 0

        # Preferred online input: MuJoCo/body IDs 55..60, confirmed by the
        # task code.  This is unambiguous and should be used in BRAND if possible.
        mapped_compact = VM_BODY_ID_TO_COMPACT_OBJECT_ID.get(raw)
        if mapped_compact is not None:
            mapped = vm_offset + mapped_compact
            if 0 <= mapped < self.n_objects:
                return mapped

        # Also accept compact VM IDs 0..5.  Six-class artifacts keep these as
        # 0..5; twelve-class pooled artifacts shift them to 6..11.
        if raw in VM_COMPACT_OBJECT_ID_TO_NAME:
            mapped = vm_offset + raw
            if 0 <= mapped < self.n_objects:
                return mapped

        # Finally, accept an already-internal model ID such as 6..11.
        if 0 <= raw < self.n_objects:
            return raw
        return None

    def _feature_row(self, neural_bin_256: np.ndarray, object_id: int | None) -> tuple[np.ndarray, float, np.ndarray | None, int | None]:
        """Build one unscaled feature row exactly matching the trained artifact."""
        phase = self._phase()
        parts = [self._causal_neural_features(neural_bin_256), np.asarray([phase], dtype=np.float32)]
        prior = None
        model_object_id = self._object_id_for_model(object_id)
        if self.model_mode == "hybrid":
            prior = self._template_at_phase(phase)
            parts.append(prior)
        if self.condition_object:
            onehot = np.zeros(self.n_objects, dtype=np.float32)
            if model_object_id is not None and 0 <= int(model_object_id) < self.n_objects:
                onehot[int(model_object_id)] = 1.0
            parts.append(onehot)
        return np.concatenate(parts).astype(np.float32), phase, prior, model_object_id

    def step(
        self,
        neural_bin_256: np.ndarray,
        *,
        decode_enabled: bool,
        timestamp_s: float,
        object_id: int | None = None,
    ) -> StreamStep | None:
        """Decode one 100 Hz bin if the external start/stop flag is active."""
        if not decode_enabled:
            if self.decode_active:
                self.reset_attempt()
            return None

        if not self.decode_active:
            self.reset_attempt()
            self.decode_active = True

        t0 = time.perf_counter()
        feature, phase, prior, model_object_id = self._feature_row(neural_bin_256, object_id)
        feature_scaled = (feature - self.x_mean) / self.x_scale
        with self.torch.no_grad():
            xb_one = self.torch.from_numpy(feature_scaled[None, None].astype(np.float32)).to(self.device)
            if self.model_family == "gru":
                # Online GRU path: keep the recurrent hidden state and process
                # only the newest 10 ms bin. This avoids recomputing the full
                # history and is the preferred first BRAND deployment path.
                h, self.rnn_hidden = self.model.gru(xb_one, self.rnn_hidden)
                pred_scaled = self.model.head(self.model.norm(h))[0, -1].detach().cpu().numpy()
            elif self.model_family == "lfads_lite":
                # LFADS-lite is also a unidirectional GRU in the deployed
                # artifact, followed by a factor bottleneck. It can therefore
                # stream one bin at a time as long as bidirectional LFADS is not
                # used for online deployment.
                h, self.rnn_hidden = self.model.gru(xb_one, self.rnn_hidden)
                factors = self.model.factor(h)
                pred_scaled = self.model.head(factors)[0, -1].detach().cpu().numpy()
            else:
                # Transformer path: causal attention still needs a context
                # window. It remains useful for offline comparison, but GRU or
                # LFADS-lite should be tested first for strict 100 Hz latency.
                self.context.append(feature_scaled.astype(np.float32))
                xb = np.stack(list(self.context), axis=0)[None]
                mask = np.ones((1, xb.shape[1]), dtype=bool)
                pred_scaled = self.model(
                    self.torch.from_numpy(xb).to(self.device),
                    self.torch.from_numpy(mask).to(self.device),
                )["y"][0, -1].detach().cpu().numpy()
        target = self._decode_target_space(pred_scaled * self.y_scale + self.y_mean)
        if self.model_mode == "hybrid":
            assert prior is not None
            target = prior + target
        qpos = self._target_to_qpos(target)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.bin_in_attempt += 1
        return StreamStep(
            timestamp_s=float(timestamp_s),
            qpos=qpos.astype(np.float32),
            phase=phase,
            latency_ms=float(latency_ms),
            object_id=model_object_id,
            raw_object_id=self.default_object_id if object_id is None else object_id,
        )


def load_template_prior(path: Path | None) -> np.ndarray | None:
    """Load an optional hybrid template prior from .npy."""
    if path is None:
        return None
    arr = np.load(path).astype(np.float32)
    if arr.shape[0] != N_QPOS and arr.shape[1] != N_QPOS:
        raise ValueError(f"Expected template prior with one 32-D axis, got {arr.shape}.")
    return arr


def _field(data: dict, name: str):
    """Return a Redis field from a bytes-keyed or str-keyed dictionary."""
    if name in data:
        return data[name]
    key = name.encode()
    if key in data:
        return data[key]
    return None


def _decode_text(value) -> str:
    """Decode bytes/string Redis values into plain text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_bool(value) -> bool:
    """Parse a decode/rest flag from bytes, numbers, or strings."""
    text = _decode_text(value).strip().lower()
    if text in {"1", "true", "decode", "active", "attempt", "go", "start"}:
        return True
    if text in {"0", "false", "rest", "break", "idle", "stop", ""}:
        return False
    try:
        return bool(float(text))
    except ValueError:
        return False


def parse_optional_int(value) -> int | None:
    """Parse optional object IDs from Redis values."""
    text = _decode_text(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float32_vector(value, *, expected: int, field_name: str) -> np.ndarray:
    """Parse a Redis vector field as raw float32 bytes or JSON/list text."""
    if value is None:
        raise ValueError(f"Missing required Redis field `{field_name}`.")
    if isinstance(value, bytes):
        arr = np.frombuffer(value, dtype=np.float32)
        if arr.size == expected:
            return arr.astype(np.float32, copy=True)
        try:
            payload = json.loads(value.decode("utf-8"))
            arr = np.asarray(payload, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise ValueError(f"Could not parse `{field_name}` as {expected} float32 values.") from exc
    else:
        arr = np.asarray(json.loads(str(value)), dtype=np.float32).reshape(-1)
    if arr.size != expected:
        raise ValueError(f"Expected `{field_name}` to contain {expected} values, got {arr.size}.")
    return arr.astype(np.float32, copy=False)


def redis_id_to_seconds(entry_id) -> float:
    """Convert Redis stream ID milliseconds into seconds when no timestamp field exists."""
    text = _decode_text(entry_id)
    try:
        ms = float(text.split("-", 1)[0])
        return ms / 1000.0
    except Exception:
        return time.time()


def publish_state(r, nickname: str, status: str, **extra) -> None:
    """Publish a BRAND-style state update to `<nickname>_state`."""
    payload = {"status": status, **{k: str(v) for k, v in extra.items()}}
    r.xadd(f"{nickname}_state", payload)


def latest_supergraph_parameters(r, nickname: str) -> dict:
    """Read this node's parameters from BRAND's `supergraph_stream`."""
    entries = r.xrevrange("supergraph_stream", max="+", min="-", count=1)
    if not entries:
        return {}
    data = entries[0][1]
    raw = _field(data, "data")
    if raw is None:
        return {}
    graph = json.loads(_decode_text(raw))
    nodes = graph.get("nodes", {})
    if isinstance(nodes, dict):
        for _key, node in nodes.items():
            if node.get("nickname") == nickname:
                return dict(node.get("parameters", {}))
    if isinstance(nodes, list):
        for node in nodes:
            if node.get("nickname") == nickname:
                return dict(node.get("parameters", {}))
    return {}


def param(params: dict, args: argparse.Namespace, name: str, default=None):
    """Return graph parameter if present, otherwise CLI/default value."""
    if name in params:
        return params[name]
    return getattr(args, name, default)


def run_brand_redis_node(args: argparse.Namespace) -> None:
    """Run the decoder as a BRAND-compatible Redis stream node."""
    try:
        import redis
    except Exception as exc:
        raise RuntimeError(
            "Python package `redis` is required for BRAND Redis mode. "
            "In the BRAND repo, run `./bootstrap.sh`, `conda activate rt`, and `make`, "
            "or install `redis` into the active environment."
        ) from exc

    if args.redis_socket:
        r = redis.Redis(unix_socket_path=args.redis_socket)
    else:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, retry_on_timeout=True)

    nickname = args.nickname or "thesis_qpos_decoder"
    publish_state(r, nickname, NODE_STARTED)
    graph_params = latest_supergraph_parameters(r, nickname)

    artifact = Path(param(graph_params, args, "artifact", args.artifact or "")).expanduser()
    if not str(artifact):
        raise ValueError("No model artifact was provided by CLI `--artifact` or graph parameter `artifact`.")

    template_prior_param = param(graph_params, args, "template_prior", args.template_prior)
    template_prior_path = Path(template_prior_param).expanduser() if template_prior_param else None
    decoder = StreamingQposDecoder(
        artifact,
        device=str(param(graph_params, args, "device", args.device)),
        sample_rate_hz=float(param(graph_params, args, "sample_rate_hz", args.sample_rate_hz)),
        expected_duration_s=float(param(graph_params, args, "expected_duration_s", args.expected_duration_s)),
        neural_smooth_alpha=float(param(graph_params, args, "neural_smooth_alpha", args.neural_smooth_alpha)),
        neural_already_smoothed=parse_bool(param(graph_params, args, "neural_already_smoothed", args.neural_already_smoothed)),
        max_context_bins=int(param(graph_params, args, "max_context_bins", args.max_context_bins)),
        default_object_id=parse_optional_int(param(graph_params, args, "object_id", args.object_id)),
        template_prior=load_template_prior(template_prior_path),
    )

    input_stream = str(param(graph_params, args, "input_stream", args.input_stream))
    flag_stream = str(param(graph_params, args, "flag_stream", args.flag_stream))
    object_stream = str(param(graph_params, args, "object_stream", args.object_stream or ""))
    output_stream = str(param(graph_params, args, "output_stream", args.output_stream))
    neural_field = str(param(graph_params, args, "neural_field", args.neural_field))
    flag_field = str(param(graph_params, args, "flag_field", args.flag_field))
    timestamp_field = str(param(graph_params, args, "timestamp_field", args.timestamp_field))
    object_field = str(param(graph_params, args, "object_field", args.object_field))
    block_ms = int(param(graph_params, args, "block_ms", args.block_ms))

    last_neural_id = str(param(graph_params, args, "redis_start_id", args.redis_start_id))
    publish_state(
        r,
        nickname,
        NODE_READY,
        artifact=str(artifact),
        input_stream=input_stream,
        flag_stream=flag_stream,
        output_stream=output_stream,
    )

    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    try:
        while running:
            reply = r.xread({input_stream: last_neural_id}, count=1, block=block_ms)
            if not reply:
                continue
            _stream, entries = reply[0]
            entry_id, data = entries[0]
            last_neural_id = _decode_text(entry_id)
            neural = parse_float32_vector(_field(data, neural_field), expected=256, field_name=neural_field)

            flag_value = _field(data, flag_field)
            if flag_value is None and flag_stream:
                flag_entries = r.xrevrange(flag_stream, max="+", min="-", count=1)
                if flag_entries:
                    flag_value = _field(flag_entries[0][1], flag_field)
            decode_enabled = parse_bool(flag_value)

            object_id = parse_optional_int(_field(data, object_field))
            if object_id is None and object_stream:
                object_entries = r.xrevrange(object_stream, max="+", min="-", count=1)
                if object_entries:
                    object_id = parse_optional_int(_field(object_entries[0][1], object_field))

            timestamp_value = _field(data, timestamp_field)
            timestamp_s = float(_decode_text(timestamp_value)) if timestamp_value is not None else redis_id_to_seconds(entry_id)
            step = decoder.step(neural, decode_enabled=decode_enabled, timestamp_s=timestamp_s, object_id=object_id)
            if step is None:
                continue
            r.xadd(
                output_stream,
                {
                    "timestamp_s": f"{step.timestamp_s:.9f}",
                    "phase": f"{step.phase:.6f}",
                    "latency_ms": f"{step.latency_ms:.6f}",
                    "object_id": "" if step.object_id is None else str(step.object_id),
                    "object_id_raw": "" if step.raw_object_id is None else str(step.raw_object_id),
                    "qpos": step.qpos.astype(np.float32).tobytes(),
                    "qpos_json": json.dumps([float(x) for x in step.qpos]),
                },
            )
    except Exception as exc:
        publish_state(r, nickname, NODE_FATAL_ERROR, error=repr(exc))
        raise
    finally:
        publish_state(r, nickname, NODE_SHUTDOWN)
        r.close()


def simulate_artificial_stream(args: argparse.Namespace) -> None:
    """Test the node with fake 256-channel bins and a decode flag window."""
    decoder = StreamingQposDecoder(
        args.artifact,
        device=args.device,
        sample_rate_hz=args.sample_rate_hz,
        expected_duration_s=args.expected_duration_s,
        neural_smooth_alpha=args.neural_smooth_alpha,
        neural_already_smoothed=args.neural_already_smoothed,
        max_context_bins=args.max_context_bins,
        default_object_id=args.object_id,
        template_prior=load_template_prior(args.template_prior),
    )
    rng = np.random.default_rng(args.seed)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_s",
                "decode_enabled",
                "phase",
                "latency_ms",
                "object_id_raw",
                "object_id_model",
                *[f"qpos_{i}" for i in range(N_QPOS)],
            ]
        )
        for i in range(args.n_bins):
            timestamp_s = i / args.sample_rate_hz
            decode_enabled = args.decode_start <= i < args.decode_stop
            fake_neural = rng.poisson(lam=args.fake_rate_hz / args.sample_rate_hz, size=256).astype(np.float32)
            step = decoder.step(fake_neural, decode_enabled=decode_enabled, timestamp_s=timestamp_s, object_id=args.object_id)
            if step is None:
                writer.writerow([timestamp_s, False, "", "", "", "", *[""] * N_QPOS])
            else:
                latencies.append(step.latency_ms)
                writer.writerow(
                    [
                        step.timestamp_s,
                        True,
                        step.phase,
                        step.latency_ms,
                        "" if step.raw_object_id is None else step.raw_object_id,
                        "" if step.object_id is None else step.object_id,
                        *step.qpos.tolist(),
                    ]
                )
    summary = {
        "artifact": str(args.artifact),
        "n_decoded_bins": len(latencies),
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "object_id_raw": args.object_id,
        "object_id_model_note": (
            "VM body IDs 55..60 are mapped to the artifact one-hot columns. "
            "Six-class strict artifacts use model IDs 0..5; older twelve-class "
            "pooled artifacts use model IDs 6..11."
        ),
        "sample_rate_hz": args.sample_rate_hz,
        "target_period_ms": 1000.0 / args.sample_rate_hz,
        "meets_100hz_median": bool(np.median(latencies) < 1000.0 / args.sample_rate_hz) if latencies else False,
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def simulate_patient_session_stream(args: argparse.Namespace) -> None:
    """Replay one local patient trial through the streaming node core."""
    from decoder_core.data import load_dataset
    from decoder_core.metrics import mean_corr, rmse, summarize

    dataset = load_dataset(args.offline_session_root)
    candidates = [
        trial
        for trial in dataset.trials
        if (args.offline_gesture is None or int(trial.gesture) == int(args.offline_gesture))
        and (args.offline_repeat is None or int(trial.repeat) == int(args.offline_repeat))
    ]
    if not candidates:
        raise ValueError(
            f"No trial matched gesture={args.offline_gesture} repeat={args.offline_repeat} "
            f"in {args.offline_session_root}."
        )
    trial = candidates[0]
    decoder = StreamingQposDecoder(
        args.artifact,
        device=args.device,
        sample_rate_hz=args.sample_rate_hz,
        expected_duration_s=trial.n_time / args.sample_rate_hz,
        neural_smooth_alpha=args.neural_smooth_alpha,
        neural_already_smoothed=args.neural_already_smoothed,
        max_context_bins=args.max_context_bins,
        default_object_id=args.object_id,
        template_prior=load_template_prior(args.template_prior),
    )

    outputs = []
    latencies = []
    # Send a short rest period first to test the start gate/reset behavior.
    for i in range(args.offline_pre_rest_bins):
        decoder.step(
            np.zeros(256, dtype=np.float32),
            decode_enabled=False,
            timestamp_s=-float(args.offline_pre_rest_bins - i) / args.sample_rate_hz,
            object_id=args.object_id if args.object_id is not None else trial.object_id,
        )
    for i, neural in enumerate(trial.neural):
        step = decoder.step(
            neural,
            decode_enabled=True,
            timestamp_s=i / args.sample_rate_hz,
            object_id=args.object_id if args.object_id is not None else trial.object_id,
        )
        assert step is not None
        outputs.append(step.qpos)
        latencies.append(step.latency_ms)
    # Send rest again to test stop/reset behavior.
    decoder.step(
        trial.neural[-1],
        decode_enabled=False,
        timestamp_s=trial.n_time / args.sample_rate_hz,
        object_id=args.object_id if args.object_id is not None else trial.object_id,
    )

    decoded = np.stack(outputs, axis=0).astype(np.float32)
    actual = trial.qpos[: decoded.shape[0]].astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "offline_actual_qpos.npy", actual)
    np.save(args.out_dir / "offline_decoded_qpos.npy", decoded)
    table = {
        "artifact": str(args.artifact),
        "session": trial.session,
        "gesture": int(trial.gesture),
        "repeat": int(trial.repeat),
        "handle_id": None if trial.handle_id is None else int(trial.handle_id),
        "object_name": trial.object_name,
        "object_id": int(trial.object_id),
        "object_id_stream_raw": args.object_id if args.object_id is not None else int(trial.object_id),
        "object_id_model_note": (
            "VM body IDs 55..60 are mapped to the artifact one-hot columns. "
            "Six-class strict artifacts use model IDs 0..5; older twelve-class "
            "pooled artifacts use model IDs 6..11."
        ),
        "n_bins": int(decoded.shape[0]),
        "sample_rate_hz": float(args.sample_rate_hz),
        "R2": float(summarize(actual, decoded)["R2"]),
        "CC": float(mean_corr(actual, decoded)),
        "RMSE": float(rmse(actual, decoded)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "target_period_ms": float(1000.0 / args.sample_rate_hz),
        "meets_100hz_median": bool(np.median(latencies) < 1000.0 / args.sample_rate_hz),
    }
    (args.out_dir / "offline_session_stream_summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    with (args.out_dir / "offline_session_stream.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp_s", "decode_enabled", "latency_ms", "object_id_raw", "object_id_model"]
            + [f"actual_qpos_{i}" for i in range(N_QPOS)]
            + [f"decoded_qpos_{i}" for i in range(N_QPOS)]
        )
        stream_object_id = args.object_id if args.object_id is not None else trial.object_id
        model_object_id = decoder._object_id_for_model(stream_object_id)
        for i, (a, d, latency) in enumerate(zip(actual, decoded, latencies)):
            writer.writerow([i / args.sample_rate_hz, True, latency, stream_object_id, model_object_id, *a.tolist(), *d.tolist()])
    print(json.dumps(table, indent=2))


def run_realtime_decode_loop(
    decoder: StreamingQposDecoder,
    *,
    read_inputs: Callable[[], tuple[np.ndarray, bool, float, int | None]],
    publish_output: Callable[[StreamStep], None],
    sample_rate_hz: float = 100.0,
    max_runtime_s: float | None = None,
) -> None:
    """Run a generic BRAND-style 100 Hz subscribe/decode/publish loop.

    `read_inputs` must return:
    - neural_bin_256: preprocessed 256-channel bin for the current loop;
    - decode_enabled: False during trial breaks/rest, True during active attempt;
    - timestamp_s: timestamp to attach to the output command;
    - object_id: optional object label for object-conditioned models.

    `publish_output` receives a StreamStep only when decode_enabled is True.
    During breaks, decoder state is reset and nothing is published.
    """
    period_s = 1.0 / float(sample_rate_hz)
    t_start = time.perf_counter()
    next_tick = t_start
    while True:
        if max_runtime_s is not None and time.perf_counter() - t_start >= max_runtime_s:
            break
        neural_bin, decode_enabled, timestamp_s, object_id = read_inputs()
        step = decoder.step(
            neural_bin,
            decode_enabled=bool(decode_enabled),
            timestamp_s=float(timestamp_s),
            object_id=object_id,
        )
        if step is not None:
            publish_output(step)
        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # If inference or upstream reads overrun 10 ms, skip sleeping and
            # let the next loop catch up. The latency_ms field should be logged
            # downstream to detect this in lab tests.
            next_tick = time.perf_counter()


def read_brand_inputs():
    """Placeholder for BRAND subscriptions.

    Replace this with the lab's actual stream reads.  Expected logical inputs:
    - a 256-D preprocessed neural bin;
    - a decode flag, e.g. 0=rest, 1=decode;
    - an optional object_id if the selected model is object-conditioned;
    - a timestamp.
    """
    raise NotImplementedError("Connect this placeholder to BRAND subscribe/read APIs.")


def publish_brand_output(step: StreamStep) -> None:
    """Placeholder for BRAND publishing of the decoded qpos command."""
    raise NotImplementedError("Connect this placeholder to BRAND publish/write APIs.")


def main() -> int:
    """CLI for artificial stream testing and latency measurement."""
    parser = argparse.ArgumentParser(description="Test or run a BRAND-style streaming qpos decoder node.")
    parser.add_argument("-n", "--nickname", type=str, default=None, help="BRAND node nickname; required by the graph launcher.")
    parser.add_argument("-i", "--redis-host", type=str, default="localhost", help="Redis host/IP for BRAND mode.")
    parser.add_argument("-p", "--redis-port", type=int, default=6379, help="Redis port for BRAND mode.")
    parser.add_argument("-s", "--redis-socket", type=str, default=None, help="Redis Unix socket for BRAND mode.")
    parser.add_argument("--brand-run", action="store_true", help="Run as a BRAND Redis stream node.")
    parser.add_argument("--artifact", type=Path, default=None, help="Saved .pt model artifact from the VM suite.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--expected-duration-s", type=float, default=4.5)
    parser.add_argument("--neural-smooth-alpha", type=float, default=0.15)
    parser.add_argument("--neural-already-smoothed", action="store_true")
    parser.add_argument("--max-context-bins", type=int, default=240)
    parser.add_argument("--object-id", type=int, default=None, help="Optional object ID for object-conditioned direct models.")
    parser.add_argument("--template-prior", type=Path, default=None, help="Optional .npy template for hybrid models.")
    parser.add_argument("--n-bins", type=int, default=600)
    parser.add_argument("--decode-start", type=int, default=100)
    parser.add_argument("--decode-stop", type=int, default=500)
    parser.add_argument("--fake-rate-hz", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out-csv", type=Path, default=Path("brand_artificial_stream_test.csv"))
    parser.add_argument("--offline-session-root", type=Path, default=None, help="Replay one local patient session folder through the node core.")
    parser.add_argument("--offline-gesture", type=int, default=None, help="Gesture index for offline session replay.")
    parser.add_argument("--offline-repeat", type=int, default=8, help="Repeat index for offline session replay.")
    parser.add_argument("--offline-pre-rest-bins", type=int, default=10, help="Rest bins sent before offline trial replay.")
    parser.add_argument("--out-dir", type=Path, default=Path("brand_offline_session_test"))
    parser.add_argument("--input-stream", default="neural_stream", help="BRAND Redis input stream carrying 256-D neural bins.")
    parser.add_argument("--flag-stream", default="decode_flag_stream", help="BRAND Redis stream carrying decode/rest flag.")
    parser.add_argument("--object-stream", default=None, help="Optional BRAND Redis stream carrying object ID.")
    parser.add_argument("--output-stream", default="decoded_qpos_stream", help="BRAND Redis output stream for decoded qpos.")
    parser.add_argument("--neural-field", default="neural", help="Redis field containing 256 float32 neural values.")
    parser.add_argument("--flag-field", default="decode", help="Redis field containing 0/1 decode flag.")
    parser.add_argument("--timestamp-field", default="timestamp_s", help="Redis field containing timestamp seconds.")
    parser.add_argument("--object-field", default="object_id", help="Redis field containing optional object ID.")
    parser.add_argument("--block-ms", type=int, default=20, help="Redis XREAD block time in milliseconds.")
    parser.add_argument("--redis-start-id", default="$", help="Redis stream ID to start reading from; `$` means new online entries only.")
    args, _unknown = parser.parse_known_args()
    if args.brand_run:
        run_brand_redis_node(args)
    elif args.offline_session_root is not None:
        if args.artifact is None:
            raise SystemExit("--artifact is required for --offline-session-root tests.")
        simulate_patient_session_stream(args)
    else:
        if args.artifact is None:
            raise SystemExit("--artifact is required unless --brand-run obtains it from supergraph_stream.")
        simulate_artificial_stream(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
