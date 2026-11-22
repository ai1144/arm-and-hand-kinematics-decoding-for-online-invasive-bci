from __future__ import annotations
"""Patient-session loading and causal feature construction.

The training suite has to support two formats:

1. legacy pickled folders with all_attempt.pkl and trials-attempt*.pkl;
2. newer folders with trials-attempt.pkl, config.yaml, and kinematics CSV files.

Only attempt trials are loaded here.  Imagine/observe files are intentionally
ignored for the decoder suite because the online target is attempted movement.
The loader always returns robot qpos/qvel/qacc/ctrl arrays in the first 32
robot dimensions and removes object freejoint qpos/qvel columns.
"""

import pickle
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from .mujoco_metadata import DEFAULT_MUJOCO_XML, parse_mujoco_metadata


SAMPLE_RATE_HZ = 100.0
DT = 1.0 / SAMPLE_RATE_HZ


def _suffix_int(name: str) -> int:
    """Extract the integer after the last underscore, e.g. qpos_12 -> 12."""
    return int(name.rsplit("_", 1)[1])


def _prefixed_index(name: str, prefix: str) -> int | None:
    """Return the numeric index after a prefix, or None when the name does not match."""
    match = re.match(rf"^{re.escape(prefix)}(\d+)(?:_|$)", str(name))
    return int(match.group(1)) if match else None


def _sorted_prefixed(columns: Iterable[str], prefix: str, limit: int | None = None) -> list[str]:
    """Select columns like qpos_0..qpos_31 in numeric order, ignoring object columns."""
    indexed_columns = []
    for col in columns:
        idx = _prefixed_index(str(col), prefix)
        if idx is None:
            continue
        if limit is not None and idx >= limit:
            continue
        indexed_columns.append((idx, col))
    return [col for _idx, col in sorted(indexed_columns, key=lambda item: item[0])]


@dataclass(slots=True)
class Trial:
    """One attempted movement repeat after neural/kinematic alignment."""

    gesture: int
    repeat: int
    neural: np.ndarray  # (T, 256), spike counts / rates
    qpos: np.ndarray  # (T, 32)
    qvel: np.ndarray  # (T, 32)
    qacc: np.ndarray  # (T, 32)
    ctrl: np.ndarray  # (T, 32), MuJoCo actuator controls; order can differ from qpos
    sensor: np.ndarray  # (T, 24), shoulder/elbow/wrist/fingertip xyz
    session: str = ""
    handle_id: int | None = None
    movement_name: str = ""
    object_name: str = ""
    object_id: int = -1
    object_code: str = ""

    @property
    def n_time(self) -> int:
        """Number of 100 Hz bins in this trial."""
        return int(self.neural.shape[0])

    def generated_time_s(self, dt: float = DT) -> np.ndarray:
        """Synthetic timestamps starting at 0 because raw timestamps were unavailable."""
        return np.round(np.arange(self.n_time, dtype=np.float64) * float(dt), 10)


@dataclass(slots=True)
class Dataset:
    """All aligned trials plus metadata needed for features and visualization."""

    trials: list[Trial]
    area_labels: list[str]
    joint_names: list[str]  # Backward-compatible alias for qpos_names.
    qpos_names: list[str]
    ctrl_names: list[str]
    sensor_columns: list[str]
    qpos_columns: list[str]
    qvel_columns: list[str]
    qacc_columns: list[str]
    ctrl_columns: list[str]


@dataclass(slots=True)
class FeatureConfig:
    """Small configuration object for causal neural feature construction."""

    channels: str = "all"
    channel_indices: tuple[int, ...] | None = None
    smooth_alpha: float | None = None
    history: int = 3
    lag: int = 0

    def label(self) -> str:
        """Human-readable label used in logs/plots for this feature setting."""
        smooth = "raw" if self.smooth_alpha is None else f"ema{self.smooth_alpha:g}"
        return f"ch={self.channels}|{smooth}|hist={self.history}|lag={self.lag}"


def load_pickle(path: Path):
    """Load a pickle while tolerating pandas-version differences in old files."""
    # Some local Python environments have pandas 1.x but the dataframes were
    # pickled by a newer pandas.  This shim restores the old unpickle hook.
    try:
        import pandas._libs.internals as internals
        from pandas.core.internals.blocks import new_block

        if not hasattr(internals, "_unpickle_block"):
            internals._unpickle_block = (
                lambda values, placement, ndim: new_block(
                    values, placement=placement, ndim=ndim
                )
            )
    except Exception:
        pass
    class ArrayCompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # NumPy 2 pickles use _core; standard numeric arrays are compatible
            # with the pinned NumPy 1 runtime used by Torch 2.2.
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)
    try:
        with path.open("rb") as f:
            return ArrayCompatibilityUnpickler(f).load()
    except (ImportError, AttributeError):
        # pandas' compatibility loader remaps old index/block class locations.
        return pd.read_pickle(path)


def default_data_root(workspace: Path | str = ".") -> Path:
    """Default local path used by older scripts when no data root is supplied."""
    return Path(workspace) / "data" / "raw" / "data and code" / "MM_sample_data_0116"


def load_area_labels(path: Path) -> list[str]:
    """Read one brain-area label per electrode channel and require 256 labels."""
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(labels) != 256:
        raise ValueError(f"Expected 256 channel area labels, got {len(labels)} from {path}")
    return labels


def load_joint_names(path: Path) -> list[str]:
    """Read joint names from an index,name CSV-like file."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        idx, name = line.split(",", 1)
        rows.append((int(idx), name.strip()))
    return [name for _, name in sorted(rows)]


def _interp_rows(values: np.ndarray, n_rows: int) -> np.ndarray:
    """Linearly resample a kinematic array to match the neural trial length."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == n_rows:
        return values.copy()
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if values.shape[0] == 1:
        return np.repeat(values, n_rows, axis=0)
    old_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, n_rows, dtype=np.float32)
    out = np.empty((n_rows, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(new_x, old_x, values[:, col]).astype(np.float32)
    return out


def _load_area_labels_for_root(root: Path) -> list[str]:
    """Find electrode-area metadata next to a session; fall back to unknown labels."""
    candidates = [
        root.parent / "AreaOfChannels.csv",
        root / "AreaOfChannels.csv",
        root.parent / "ElectrodesInfo.csv",
        root / "ElectrodesInfo.csv",
    ]
    for path in candidates:
        if path.exists():
            return load_area_labels(path)
    return ["unknown"] * 256


def _load_joint_names_for_root(root: Path, fallback: list[str]) -> list[str]:
    """Find joint-name metadata next to a session; otherwise use fallback names."""
    candidates = [root.parent / "joint_names.csv", root / "joint_names.csv"]
    for path in candidates:
        if path.exists():
            return load_joint_names(path)
    return fallback


ALL_ATTEMPT_CANDIDATES = (
    "all-attempt.pkl",
    "all_attempt.pkl",
    "all-attempt.pickle",
    "all_attempt.pickle",
)
TRIAL_ATTEMPT_CANDIDATES = (
    "trials-attempt-resampled.pkl",
    "trials_attempt_resampled.pkl",
    "trials-attempt-resample.pkl",
    "trials_attempt_resample.pkl",
    "trials-attempt.pkl",
    "trials_attempt.pkl",
    "trial-attempt.pkl",
    "trial_attempt.pkl",
)

LEGACY_VM_OBJECT_TO_ID = {
    "small_ball": 0,
    "big_ball": 1,
    "small_v_cylinder": 2,
    "big_v_cylinder": 3,
    "small_h_cylinder": 4,
    "big_h_cylinder": 5,
}

# Confirmed body-id mapping for the newer six-object VM sessions.
# These IDs describe the object bodies in the MuJoCo scene.  The decoder itself
# does not use body IDs directly; we use them to make object names explicit and
# consistent when the VM config is missing or hard to locate.
VM_BODY_ID_TO_OBJECT = {
    55: "small_ball",
    56: "big_ball",
    57: "small_v_cylinder",
    58: "big_v_cylinder",
    59: "small_h_cylinder",
    60: "big_h_cylinder",
}
VM_OBJECT_NAME_ALIASES = {
    "small ball": "small_ball",
    "big ball": "big_ball",
    "small vertical cylinder": "small_v_cylinder",
    "big vertical cylinder": "big_v_cylinder",
    "small horizontal cylinder": "small_h_cylinder",
    "big horizontal cylinder": "big_h_cylinder",
    "small_v_cylinder": "small_v_cylinder",
    "big_v_cylinder": "big_v_cylinder",
    "small_h_cylinder": "small_h_cylinder",
    "big_h_cylinder": "big_h_cylinder",
    "small_ball": "small_ball",
    "big_ball": "big_ball",
}

# Same-threshold VM protocol mapping received for the 2026-09 rerun.
#
# The pickle files do not store an explicit Object.Id per attempted trajectory.
# Instead, the experiment defines two linked identifiers:
#   1. handle_id / handle_inx: the task/gesture handle number used in the trial list;
#   2. geom_id: the MuJoCo object/body ID that was grasped for that handle.
#
# These constants map handle -> geom -> canonical object name.  The date-specific
# sequences below describe the order in which handles appear in all_attempt.pkl /
# trials-attempt-resampled.pkl.  For folders named "...-block1", "...-block2",
# or "...-block3", the relevant six-handle slice is used automatically.
PROTOCOL_HANDLE_TO_GEOM_ID = dict(
    zip(
        [82, 400, 57, 398, 7, 80, 90, 410, 33, 409, 24, 87, 98, 436, 47, 422, 41, 95],
        [55, 57, 60, 58, 59, 56, 55, 57, 60, 58, 59, 56, 55, 57, 60, 58, 59, 56],
        strict=True,
    )
)

SAME_THRESHOLD_HANDLE_ORDER_BY_DATE = {
    "20260707": [7, 80, 400, 57, 82, 398, 87, 33, 409, 410, 90, 24, 422, 436, 41, 98, 47, 95],
    "20260703": [398, 57, 80, 82, 7, 400, 87, 90, 409, 24, 410, 33, 95, 41, 436, 98, 47, 422],
    "20260630": [398, 82, 57, 80, 7, 400, 409, 87, 410, 33, 90, 24, 41, 436, 422, 98, 95, 47],
}


def _session_date_key(name: str) -> str:
    """Return an eight-digit session date such as 20260703 from a folder name."""
    match = re.search(r"(20\d{6})", str(name))
    return match.group(1) if match else ""


def _session_block_index(name: str) -> int | None:
    """Return block number from names like 20260703-block2, otherwise None."""
    match = re.search(r"block[-_ ]*(\d+)", str(name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _same_threshold_handle_order_for_root(root: Path, n_items: int) -> list[int] | None:
    """Return true handle IDs for the same-threshold VM files when known.

    The date-level protocol order has 18 handles, i.e. three groups of six.
    Some VM folders contain all 18 groups together; other folders contain one
    block with only six groups.  This helper supports both layouts.
    """
    date_key = _session_date_key(root.name)
    full_order = SAME_THRESHOLD_HANDLE_ORDER_BY_DATE.get(date_key)
    if full_order is None:
        return None
    if n_items == len(full_order):
        return list(full_order)
    block_index = _session_block_index(root.name)
    if n_items == 6 and block_index in {1, 2, 3}:
        start = (int(block_index) - 1) * 6
        return list(full_order[start : start + 6])
    return None


def _object_order_from_handle_order(handle_order: list[int]) -> list[str] | None:
    """Convert handle IDs into canonical object names via geom/body IDs."""
    names: list[str] = []
    for handle_id in handle_order:
        geom_id = PROTOCOL_HANDLE_TO_GEOM_ID.get(int(handle_id))
        object_name = VM_BODY_ID_TO_OBJECT.get(int(geom_id)) if geom_id is not None else None
        if object_name not in LEGACY_VM_OBJECT_TO_ID:
            return None
        names.append(str(object_name))
    return names


def _normal_pickle_name(name: str) -> str:
    """Normalize pickle names so dash/underscore variants are equivalent."""
    return name.lower().replace("_", "-")


def _find_named_pickle(root: Path, candidates: tuple[str, ...]) -> Path | None:
    """Find an expected attempt pickle and ignore imagine/observe/flag files."""
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    wanted = {_normal_pickle_name(name) for name in candidates}
    for child in root.glob("*.p*"):
        if not child.is_file():
            continue
        normal = _normal_pickle_name(child.name)
        if "flag" in normal or "imagine" in normal or "observe" in normal:
            continue
        if normal in wanted:
            return child
    return None


def _legacy_object_order_from_columns(columns: Iterable[str]) -> list[str]:
    """Infer object order from legacy VM kinematic columns.

    The new VM blocks may not have config.yaml, but their all_attempt dataframes
    expose object freejoint columns after the first 32 robot qpos columns, for
    example qpos_46_small_ball_freejoint.  When there are six all_attempt
    trajectories and six such object groups, the gesture index can be mapped to
    this ordered object list.  This gives the new-only run meaningful per-object
    plots without needing the old config.yaml format.
    """
    object_groups: list[tuple[int, str]] = []
    seen: set[str] = set()
    for col in columns:
        match = re.match(r"^qpos_(\d+)_(.+)_freejoint$", str(col))
        if not match:
            continue
        idx = int(match.group(1))
        name = match.group(2)
        if idx < 32 or name in seen:
            continue
        object_groups.append((idx, name))
        seen.add(name)
    return [name for _idx, name in sorted(object_groups, key=lambda item: item[0])]


def _canonical_vm_object_name(value: object) -> str:
    """Normalize VM object labels into the six canonical six-object names."""
    text = str(value).strip()
    if not text:
        return ""
    # Config files sometimes encode body IDs instead of names.  Accept either.
    try:
        as_int = int(float(text))
        if as_int in VM_BODY_ID_TO_OBJECT:
            return VM_BODY_ID_TO_OBJECT[as_int]
    except ValueError:
        pass
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return VM_OBJECT_NAME_ALIASES.get(normalized, normalized)


def _object_groups_from_qpos_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    """Return object-name -> qpos freejoint columns for VM object bodies."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for col in columns:
        match = re.match(r"^qpos_(\d+)_(.+)_freejoint$", str(col))
        if not match:
            continue
        idx = int(match.group(1))
        if idx < 32:
            continue
        name = _canonical_vm_object_name(match.group(2))
        if name in LEGACY_VM_OBJECT_TO_ID:
            groups.setdefault(name, []).append((idx, str(col)))
    return {
        name: [col for _idx, col in sorted(items, key=lambda item: item[0])]
        for name, items in groups.items()
    }


def _config_candidates_for_root(root: Path) -> list[Path]:
    """List possible VM config paths, including an optional environment override."""
    candidates: list[Path] = []
    env_path = os.environ.get("THESIS_VM_CONFIG_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            root / "config.yaml",
            root / "config.yml",
            root.parent / "config.yaml",
            root.parent / "config.yml",
            root.parent.parent / "config.yaml",
            root.parent.parent / "config.yml",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _objects_from_yaml_payload(payload: object, session_name: str, n_items: int) -> list[str] | None:
    """Best-effort extraction of one object order list from an unknown config shape."""
    if isinstance(payload, dict):
        # Prefer an exact session/block entry when configs contain several
        # blocks in one file.
        for key in (session_name, session_name.replace("-", "_"), session_name.replace("_", "-")):
            child = payload.get(key)
            found = _objects_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found

        exp = payload.get("experiment", payload)
        if isinstance(exp, dict):
            handles = exp.get("handle_inx") or exp.get("handle_ids") or exp.get("handles")
            geom_ids = exp.get("geom_ids") or exp.get("body_ids") or exp.get("body_inx") or exp.get("body_indices")
            if isinstance(handles, list) and isinstance(geom_ids, list) and len(handles) == len(geom_ids) == n_items:
                # Configs can define handle_id -> geom_id directly.  This is
                # the most explicit object mapping because geom IDs 55..60 are
                # the physical objects in the MuJoCo scene.
                canonical = [_canonical_vm_object_name(x) for x in geom_ids]
                if all(name in LEGACY_VM_OBJECT_TO_ID for name in canonical):
                    return canonical

            names = (
                exp.get("objects")
                or exp.get("object_names")
                or exp.get("objects_ids")
                or exp.get("object_ids")
                or exp.get("body_ids")
                or exp.get("body_inx")
                or exp.get("body_indices")
            )
            if isinstance(names, list) and len(names) == n_items:
                canonical = [_canonical_vm_object_name(x) for x in names]
                if all(name in LEGACY_VM_OBJECT_TO_ID for name in canonical):
                    return canonical

        # Recursive fallback for nested files with a different top-level layout.
        for child in payload.values():
            found = _objects_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found

    if isinstance(payload, list):
        # Some configs are a list of per-block dictionaries.
        for child in payload:
            found = _objects_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found

    return None


def _handles_from_yaml_payload(payload: object, session_name: str, n_items: int) -> list[int] | None:
    """Best-effort extraction of one handle-order list from an unknown config shape."""
    if isinstance(payload, dict):
        for key in (session_name, session_name.replace("-", "_"), session_name.replace("_", "-")):
            child = payload.get(key)
            found = _handles_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found

        exp = payload.get("experiment", payload)
        if isinstance(exp, dict):
            handles = exp.get("handle_inx") or exp.get("handle_ids") or exp.get("handles")
            if isinstance(handles, list) and len(handles) == n_items:
                try:
                    return [int(float(x)) for x in handles]
                except Exception:
                    pass

        for child in payload.values():
            found = _handles_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found

    if isinstance(payload, list):
        for child in payload:
            found = _handles_from_yaml_payload(child, session_name, n_items)
            if found is not None:
                return found
    return None


def _legacy_handle_order_from_config(root: Path, n_items: int) -> list[int] | None:
    """Read handle order from override, same-threshold constants, or config YAML."""
    override = os.environ.get("THESIS_VM_HANDLE_ORDER_JSON", "").strip()
    if override:
        payload = json.loads(override)
        if isinstance(payload, dict):
            order = payload.get(root.name, payload.get(_session_date_key(root.name), payload.get("default")))
        else:
            order = payload
        if isinstance(order, list) and len(order) == n_items:
            return [int(float(x)) for x in order]
        raise ValueError(
            "THESIS_VM_HANDLE_ORDER_JSON was set but did not contain a valid "
            f"{n_items}-handle order for {root.name}."
        )

    same_threshold_order = _same_threshold_handle_order_for_root(root, n_items)
    if same_threshold_order is not None:
        return same_threshold_order

    for path in _config_candidates_for_root(root):
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text())
        except Exception:
            continue
        found = _handles_from_yaml_payload(payload, root.name, n_items)
        if found is not None:
            return found
    return None


def _legacy_object_order_from_config(root: Path, n_items: int) -> list[str] | None:
    """Read object order from a nearby or explicitly provided VM config file."""
    override = os.environ.get("THESIS_VM_OBJECT_ORDER_JSON", "").strip()
    if override:
        # preferred corrected-vm path:
        #
        # the new vm pickle files do not store an explicit object id, and the
        # six kinematic groups appear in different orders across blocks. this
        # json tells the loader the true group order for each block, so all ten
        # neural repetitions of group 0, group 1, ... receive the right label.
        #
        # example:
        # THESIS_VM_OBJECT_ORDER_JSON='{"20260623-block1":["small_ball",...],"default":[...]}'
        payload = json.loads(override)
        if isinstance(payload, dict):
            order = payload.get(root.name, payload.get("default"))
        else:
            order = payload
        if isinstance(order, list) and len(order) == n_items:
            canonical = [_canonical_vm_object_name(x) for x in order]
            if all(name in LEGACY_VM_OBJECT_TO_ID for name in canonical):
                return canonical
        raise ValueError(
            "THESIS_VM_OBJECT_ORDER_JSON was set but did not contain a valid "
            f"{n_items}-object order for {root.name}."
        )

    handle_order = _legacy_handle_order_from_config(root, n_items)
    if handle_order is not None:
        object_order = _object_order_from_handle_order(handle_order)
        if object_order is not None:
            return object_order

    for path in _config_candidates_for_root(root):
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text())
        except Exception:
            continue
        found = _objects_from_yaml_payload(payload, root.name, n_items)
        if found is not None:
            return found
    return None


def _infer_active_object_from_kinematics(kin: pd.DataFrame) -> str:
    """Infer the grasped VM object from final hand/sensor distance to object bodies."""
    object_groups = _object_groups_from_qpos_columns(kin.columns)
    if not object_groups:
        return ""

    # fallback only:
    #
    # if an object freejoint actually moves during the logged trajectory, the
    # moving object is likely the manipulated object. this is useful for
    # inspection, but the corrected 2026-09-01 result did not rely on this
    # because inactive object coordinates could contain huge invalid values.
    movement_scores: dict[str, float] = {}
    for name, cols in object_groups.items():
        xyz_cols = cols[:3]
        if len(xyz_cols) < 3 or not all(c in kin.columns for c in xyz_cols):
            continue
        xyz = kin[xyz_cols].to_numpy(dtype=np.float32)
        movement_scores[name] = float(np.linalg.norm(xyz[-1] - xyz[0]))
    if movement_scores and max(movement_scores.values()) > 1e-4:
        return max(movement_scores, key=movement_scores.get)

    # second fallback only:
    #
    # if objects are static, the hand/fingertips should end up closest to the
    # target object at grasp time. the sensor columns are xyz triplets, so the
    # final sensor points are compared with each object's final xyz position.
    sensor_cols = _sorted_prefixed(kin.columns, "sensor_", limit=24)
    if len(sensor_cols) >= 3:
        final_sensor = kin[sensor_cols].to_numpy(dtype=np.float32)[-1]
        sensor_points = final_sensor[: (len(final_sensor) // 3) * 3].reshape(-1, 3)
        distance_scores: dict[str, float] = {}
        for name, cols in object_groups.items():
            xyz_cols = cols[:3]
            if len(xyz_cols) < 3 or not all(c in kin.columns for c in xyz_cols):
                continue
            obj_pos = kin[xyz_cols].to_numpy(dtype=np.float32)[-1]
            distance_scores[name] = float(np.min(np.linalg.norm(sensor_points - obj_pos[None, :], axis=1)))
        if distance_scores:
            return min(distance_scores, key=distance_scores.get)

    return ""


def _legacy_object_labels_for_all_attempt(root: Path, all_attempt: list[pd.DataFrame]) -> list[tuple[str, int, str]]:
    """Assign one object label to each VM all_attempt trajectory.

    Priority:
    1. explicit THESIS_VM_OBJECT_ORDER_JSON override;
    2. nearby / explicitly pointed config.yaml;
    3. per-trajectory kinematic inference from object freejoint/sensor geometry,
       only after inspection confirms it is safe for the current files;
    4. old column-order fallback;
    5. unknown.
    """
    n_items = len(all_attempt)
    config_order = _legacy_object_order_from_config(root, n_items)
    if config_order is not None:
        if os.environ.get("THESIS_VM_OBJECT_ORDER_JSON", "").strip():
            source = "object_order_override"
        elif os.environ.get("THESIS_VM_HANDLE_ORDER_JSON", "").strip():
            source = "handle_order_override_to_geom"
        elif _same_threshold_handle_order_for_root(root, n_items) is not None:
            source = "same_threshold_handle_to_geom"
        else:
            source = "config"
        return [(name, LEGACY_VM_OBJECT_TO_ID[name], source) for name in config_order]

    inferred = [_infer_active_object_from_kinematics(kin) for kin in all_attempt]
    if all(name in LEGACY_VM_OBJECT_TO_ID for name in inferred) and len(set(inferred)) > 1:
        return [(name, LEGACY_VM_OBJECT_TO_ID[name], "kinematic_geometry_inferred") for name in inferred]

    require_confident = os.environ.get("THESIS_VM_REQUIRE_CONFIDENT_OBJECT_LABELS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if require_confident:
        raise ValueError(
            "Could not assign confident object labels from config/override or kinematic geometry for "
            f"{root}. Because object order can differ across VM blocks, column-order fallback was disabled. "
            "Provide THESIS_VM_CONFIG_PATH or THESIS_VM_OBJECT_ORDER_JSON if geometry inference is not available."
        )

    column_order = _legacy_object_order_from_columns(all_attempt[0].columns)
    if (
        len(column_order) > 0
        and len(all_attempt) == len(column_order)
        and all(name in LEGACY_VM_OBJECT_TO_ID for name in column_order)
    ):
        return [(name, LEGACY_VM_OBJECT_TO_ID[name], "column_order_fallback") for name in column_order]

    return [("", -1, "unknown") for _ in all_attempt]


def _neural_trial_to_time_rows(raw_rep):
    a = np.asarray(raw_rep, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Neural trial must be 2-D; found {a.shape}")
    # Source indices 0..255 only. Exclude undocumented final two entries.
    if a.shape[0] in (256, 258):
        return a[:256].T
    if a.shape[1] in (256, 258):
        return a[:, :256]
    raise ValueError(f"Unsupported input shape {a.shape}; expected 256 or 258 source features")


def _load_dataset_legacy(root: Path, mujoco_xml: Path | str | None) -> Dataset:
    """Load legacy pickle-only sessions such as 0116 and the VM six-object blocks."""
    trial_pickle = _find_named_pickle(root, TRIAL_ATTEMPT_CANDIDATES)
    all_attempt_pickle = _find_named_pickle(root, ALL_ATTEMPT_CANDIDATES)
    if trial_pickle is None or all_attempt_pickle is None:
        raise FileNotFoundError(f"Could not find attempt pickles under {root}.")
    trials_attempt = load_pickle(trial_pickle)
    all_attempt: list[pd.DataFrame] = load_pickle(all_attempt_pickle)

    qpos_cols = _sorted_prefixed(all_attempt[0].columns, "qpos_", limit=32)
    qvel_cols = _sorted_prefixed(all_attempt[0].columns, "qvel_", limit=32)
    qacc_cols = _sorted_prefixed(all_attempt[0].columns, "qacc_", limit=32)
    ctrl_cols = _sorted_prefixed(all_attempt[0].columns, "ctrl_", limit=32)
    sensor_cols = _sorted_prefixed(all_attempt[0].columns, "sensor_", limit=24)
    # Legacy 0116-style data did not always carry explicit object labels.  The
    # six-object VM blocks do contain object bodies, and sometimes a config file
    # elsewhere on the VM.  This helper assigns one object label per trajectory
    # without assuming that object order is identical across blocks/sessions.
    object_labels = _legacy_object_labels_for_all_attempt(root, all_attempt)
    handle_order = _legacy_handle_order_from_config(root, len(all_attempt))

    if len(qpos_cols) != 32 or len(qvel_cols) != 32 or len(ctrl_cols) != 32:
        raise ValueError(
            "Expected 32 robot qpos, qvel, and ctrl columns after removing object free-joints; "
            f"got qpos={len(qpos_cols)}, qvel={len(qvel_cols)}, ctrl={len(ctrl_cols)}."
        )
    if qacc_cols and len(qacc_cols) != 32:
        raise ValueError(f"Expected either 0 or 32 robot qacc columns; got {len(qacc_cols)}.")

    trials: list[Trial] = []
    for gesture, reps in enumerate(trials_attempt):
        kin = all_attempt[gesture]
        # Repeat-level spike trials inherit the object label assigned to the
        # matching all_attempt kinematic trajectory.
        object_name, object_id, object_label_source = object_labels[gesture]
        handle_id = int(handle_order[gesture]) if handle_order is not None else int(gesture)
        qpos = kin[qpos_cols].to_numpy(dtype=np.float32)
        qvel = kin[qvel_cols].to_numpy(dtype=np.float32)
        if qacc_cols:
            qacc = kin[qacc_cols].to_numpy(dtype=np.float32)
        else:
            qacc = np.gradient(qvel, DT, axis=0).astype(np.float32)
        ctrl = kin[ctrl_cols].to_numpy(dtype=np.float32)
        sensor = (
            kin[sensor_cols].to_numpy(dtype=np.float32)
            if len(sensor_cols) == 24
            else np.zeros((qpos.shape[0], 24), dtype=np.float32)
        )
        for repeat, raw_rep in enumerate(reps):
            neural = _neural_trial_to_time_rows(raw_rep)
            t = min(neural.shape[0], qpos.shape[0])
            if t <= 2:
                raise ValueError(f"Trial {gesture}/{repeat} is too short: {t}")
            trials.append(
                Trial(
                    gesture=gesture,
                    repeat=repeat,
                    neural=neural[:t].copy(),
                    qpos=qpos[:t].copy(),
                    qvel=qvel[:t].copy(),
                    qacc=qacc[:t].copy(),
                    ctrl=ctrl[:t].copy(),
                    sensor=sensor[:t].copy(),
                    session=root.name,
                    handle_id=handle_id,
                    movement_name=f"{object_name or 'gesture'}_handle_{handle_id}",
                    object_name=object_name,
                    object_id=object_id,
                    object_code=object_label_source,
                )
            )

    labels = _load_area_labels_for_root(root)
    fallback_names = [f"joint_{i}" for i in range(32)]
    csv_joint_names = _load_joint_names_for_root(root, fallback_names)
    metadata = parse_mujoco_metadata(Path(mujoco_xml)) if mujoco_xml is not None else None
    if metadata and len(metadata.qpos_names) == len(qpos_cols) and len(metadata.ctrl_names) == len(ctrl_cols):
        qpos_names = metadata.qpos_names
        ctrl_names = metadata.ctrl_names
    else:
        qpos_names = csv_joint_names
        ctrl_names = csv_joint_names
    return Dataset(
        trials=trials,
        area_labels=labels,
        joint_names=qpos_names,
        qpos_names=qpos_names,
        ctrl_names=ctrl_names,
        sensor_columns=sensor_cols,
        qpos_columns=qpos_cols,
        qvel_columns=qvel_cols,
        qacc_columns=qacc_cols,
        ctrl_columns=ctrl_cols,
    )


def _load_dataset_csv_kinematics(root: Path, mujoco_xml: Path | str | None) -> Dataset:
    """Load sessions with spike pickle plus config.yaml and kinematics CSV files."""
    trial_pickle = _find_named_pickle(root, TRIAL_ATTEMPT_CANDIDATES)
    if trial_pickle is None:
        raise FileNotFoundError(f"Could not find trials-attempt pickle under {root}.")
    spike_trials = load_pickle(trial_pickle)
    config = yaml.safe_load((root / "config.yaml").read_text())
    exp_cfg = config["experiment"]
    handles = [int(x) for x in exp_cfg["handle_inx"]]
    n_attempts = int(config["experiment"].get("No_attempts", 10))
    expected = len(handles) * n_attempts
    if len(spike_trials) != expected:
        raise ValueError(
            f"Expected {expected} attempt trials from {len(handles)} handles x {n_attempts} attempts, "
            f"got {len(spike_trials)} in {trial_pickle}."
        )

    first_csv = root / "kinematics" / f"simulation_log_{handles[0]}.csv"
    first_df = pd.read_csv(first_csv, nrows=1)
    qpos_cols = _sorted_prefixed(first_df.columns, "qpos_", limit=32)
    qvel_cols = _sorted_prefixed(first_df.columns, "qvel_", limit=32)
    qacc_cols = _sorted_prefixed(first_df.columns, "qacc_", limit=32)
    ctrl_cols = _sorted_prefixed(first_df.columns, "ctrl_", limit=32)
    if len(qpos_cols) != 32 or len(qvel_cols) != 32 or len(ctrl_cols) != 32:
        raise ValueError(f"Expected 32 qpos/qvel/ctrl columns in {first_csv}.")
    if not qacc_cols:
        qacc_cols = [f"qacc_{i}" for i in range(32)]
    sensor_cols = [f"sensor_{i}" for i in range(24)]

    kin_cache: dict[int, pd.DataFrame] = {}

    # Newer local sessions store exact handle/object mapping in config.yaml.
    # The odd spellings are kept unchanged because they are source-data labels
    # and downstream tables should be traceable back to the raw config.
    object_names_cfg = exp_cfg.get("objects") or exp_cfg.get("object_names") or []
    object_codes_cfg = exp_cfg.get("objects_ids") or exp_cfg.get("object_ids") or []
    canonical_object_order = [
        "small_ring",
        "big_horizontal_cylindar",
        "big_vertical_cylindar",
        "samll_horizontal_cylindar",
        "handle",
        "small_vertical_cylindar",
    ]
    known_object_rows = [
        (524, "21.1", "small_ring"),
        (5, "56.1", "big_horizontal_cylindar"),
        (398, "76.1", "big_vertical_cylindar"),
        (7, "51.1", "samll_horizontal_cylindar"),
        (744, "0.1", "handle"),
        (400, "71.1", "small_vertical_cylindar"),
        (533, "21.2", "small_ring"),
        (10, "56.2", "big_horizontal_cylindar"),
        (409, "76.2", "big_vertical_cylindar"),
        (24, "51.2", "samll_horizontal_cylindar"),
        (743, "0.2", "handle"),
        (410, "71.2", "small_vertical_cylindar"),
        (538, "21.3", "small_ring"),
        (47, "56.3", "big_horizontal_cylindar"),
        (422, "76.3", "big_vertical_cylindar"),
        (41, "51.3", "samll_horizontal_cylindar"),
        (742, "0.3", "handle"),
        (436, "71.3", "small_vertical_cylindar"),
    ]
    fallback_object_map = {handle: (code, name) for handle, code, name in known_object_rows}

    def object_map_from_experiment(exp: dict) -> dict[int, tuple[str, str]]:
        """Map handle_id -> (object_code, object_name) from config.yaml lists."""
        ref_handles = [int(x) for x in exp.get("handle_inx", [])]
        ref_names = list(exp.get("objects", []) or exp.get("object_names", []) or [])
        ref_codes = list(exp.get("objects_ids", []) or exp.get("object_ids", []) or [])
        if len(ref_handles) == 0 or len(ref_names) != len(ref_handles):
            return {}
        return {
            int(handle_id): (
                str(ref_codes[i]) if i < len(ref_codes) else "",
                str(ref_names[i]),
            )
            for i, handle_id in enumerate(ref_handles)
        }

    parent_config_map: dict[int, tuple[str, str]] = {}
    parent_config = root.parent / "config.yaml"
    if parent_config.exists():
        try:
            parent_exp = yaml.safe_load(parent_config.read_text()).get("experiment", {})
            parent_config_map = object_map_from_experiment(parent_exp)
        except Exception:
            parent_config_map = {}

    config_object_map: dict[int, tuple[str, str]] = {}
    if len(object_names_cfg) == len(handles):
        for i, handle_id in enumerate(handles):
            code = str(object_codes_cfg[i]) if i < len(object_codes_cfg) else ""
            config_object_map[int(handle_id)] = (code, str(object_names_cfg[i]))
    object_name_to_id = {name: i for i, name in enumerate(canonical_object_order)}

    def get_kin(handle_id: int) -> pd.DataFrame:
        """Load one handle's kinematic CSV once and cache it for all repeats."""
        if handle_id not in kin_cache:
            kin_cache[handle_id] = pd.read_csv(root / "kinematics" / f"simulation_log_{handle_id}.csv")
        return kin_cache[handle_id]

    trials: list[Trial] = []
    for gesture, handle_id in enumerate(handles):
        object_code, object_name = config_object_map.get(
            handle_id,
            parent_config_map.get(handle_id, fallback_object_map.get(handle_id, ("", ""))),
        )
        object_id = object_name_to_id.get(object_name, -1)
        kin = get_kin(handle_id)
        qpos_hi = kin[qpos_cols].to_numpy(dtype=np.float32)
        qvel_hi = kin[qvel_cols].to_numpy(dtype=np.float32)
        ctrl_hi = kin[ctrl_cols].to_numpy(dtype=np.float32)
        qacc_hi = kin[qacc_cols].to_numpy(dtype=np.float32) if all(c in kin.columns for c in qacc_cols) else None
        for repeat in range(n_attempts):
            raw_rep = spike_trials[gesture * n_attempts + repeat]
            neural = _neural_trial_to_time_rows(raw_rep)
            t = int(neural.shape[0])
            if t <= 2:
                raise ValueError(f"Trial handle {handle_id}/repeat {repeat} is too short: {t}")
            qpos = _interp_rows(qpos_hi, t)
            qvel = _interp_rows(qvel_hi, t)
            ctrl = _interp_rows(ctrl_hi, t)
            if qacc_hi is not None:
                qacc = _interp_rows(qacc_hi, t)
            else:
                qacc = np.gradient(qvel, DT, axis=0).astype(np.float32)
            sensor = np.zeros((t, 24), dtype=np.float32)
            trials.append(
                Trial(
                    gesture=gesture,
                    repeat=repeat,
                    neural=neural.copy(),
                    qpos=qpos,
                    qvel=qvel,
                    qacc=qacc,
                    ctrl=ctrl,
                    sensor=sensor,
                    session=root.name,
                    handle_id=handle_id,
                    movement_name=f"handle_{handle_id}",
                    object_name=object_name,
                    object_id=object_id,
                    object_code=object_code,
                )
            )

    metadata = parse_mujoco_metadata(Path(mujoco_xml)) if mujoco_xml is not None else None
    fallback_names = [f"joint_{i}" for i in range(32)]
    csv_joint_names = _load_joint_names_for_root(root, fallback_names)
    if metadata and len(metadata.qpos_names) == len(qpos_cols) and len(metadata.ctrl_names) == len(ctrl_cols):
        qpos_names = metadata.qpos_names
        ctrl_names = metadata.ctrl_names
    else:
        qpos_names = csv_joint_names
        ctrl_names = csv_joint_names

    return Dataset(
        trials=trials,
        area_labels=_load_area_labels_for_root(root),
        joint_names=qpos_names,
        qpos_names=qpos_names,
        ctrl_names=ctrl_names,
        sensor_columns=sensor_cols,
        qpos_columns=qpos_cols,
        qvel_columns=qvel_cols,
        qacc_columns=qacc_cols,
        ctrl_columns=ctrl_cols,
    )


def load_dataset(data_root: Path | str | None = None, mujoco_xml: Path | str | None = DEFAULT_MUJOCO_XML) -> Dataset:
    """Auto-detect a supported patient-session layout and return aligned trials."""
    root = Path(data_root) if data_root is not None else default_data_root(Path.cwd())
    trial_pickle = _find_named_pickle(root, TRIAL_ATTEMPT_CANDIDATES)
    all_attempt_pickle = _find_named_pickle(root, ALL_ATTEMPT_CANDIDATES)
    if trial_pickle is not None and all_attempt_pickle is not None:
        return _load_dataset_legacy(root, mujoco_xml)
    if trial_pickle is not None and (root / "config.yaml").exists() and (root / "kinematics").exists():
        return _load_dataset_csv_kinematics(root, mujoco_xml)
    raise FileNotFoundError(f"Could not recognize dataset layout under {root}.")


def split_trials(
    trials: Iterable[Trial],
    train_repeats: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    val_repeats: tuple[int, ...] = (6, 7),
    test_repeats: tuple[int, ...] = (8, 9),
) -> tuple[list[Trial], list[Trial], list[Trial]]:
    """Default repeat-holdout split: early repeats train, middle val, late test."""
    train, val, test = [], [], []
    for trial in trials:
        if trial.repeat in train_repeats:
            train.append(trial)
        elif trial.repeat in val_repeats:
            val.append(trial)
        elif trial.repeat in test_repeats:
            test.append(trial)
    return train, val, test


def resolve_channels(dataset: Dataset, cfg: FeatureConfig) -> np.ndarray:
    """Convert a channel selector such as all/motor/no_vM1 into electrode indices."""
    if cfg.channel_indices is not None:
        idx = np.asarray(cfg.channel_indices, dtype=np.int64)
    elif cfg.channels == "all":
        idx = np.arange(256, dtype=np.int64)
    elif cfg.channels == "motor":
        idx = np.asarray(
            [i for i, area in enumerate(dataset.area_labels) if area in {"vM1", "dM1"}],
            dtype=np.int64,
        )
    elif cfg.channels == "nonmotor":
        idx = np.asarray(
            [i for i, area in enumerate(dataset.area_labels) if area in {"ppC", "pMC"}],
            dtype=np.int64,
        )
    elif cfg.channels.startswith("no_"):
        drop = cfg.channels[3:]
        idx = np.asarray([i for i, area in enumerate(dataset.area_labels) if area != drop], dtype=np.int64)
    else:
        idx = np.asarray([i for i, area in enumerate(dataset.area_labels) if area == cfg.channels], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"Unknown channel selector: {cfg.channels}")
    if idx.min(initial=0) < 0 or idx.max(initial=0) >= 256:
        raise ValueError(f"Channel indices out of range: {idx}")
    return idx


def causal_ema(x: np.ndarray, alpha: float | None) -> np.ndarray:
    """Apply causal exponential smoothing so each bin uses only current/past data."""
    if alpha is None:
        return np.asarray(x, dtype=np.float32)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"smooth_alpha must be in (0, 1], got {alpha}")
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    old = np.float32(1.0 - alpha)
    new = np.float32(alpha)
    for t in range(1, x.shape[0]):
        out[t] = new * x[t] + old * out[t - 1]
    return out


def build_features_for_trial(trial: Trial, channels: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Build the neural feature matrix for one trial with optional history taps."""
    # Features are causal: at bin t the model can use the current neural bin and
    # a fixed number of previous bins, never future neural activity.
    neural = causal_ema(trial.neural[:, channels], cfg.smooth_alpha)
    history = int(cfg.history)
    if history < 1:
        raise ValueError("history must be >= 1")
    t, c = neural.shape
    feats = np.zeros((t, c * history), dtype=np.float32)
    for tap in range(history):
        start = tap * c
        end = start + c
        if tap == 0:
            feats[:, start:end] = neural
        else:
            feats[tap:, start:end] = neural[:-tap]
    return feats


def aligned_target(trial: Trial, target_mode: str, lag: int) -> np.ndarray:
    """Return qpos/qvel/qpos+qvel/sensor/ctrl target aligned to the chosen lag."""
    if target_mode == "qpos":
        y = trial.qpos
    elif target_mode == "qvel":
        y = trial.qvel
    elif target_mode == "pv":
        y = np.concatenate([trial.qpos, trial.qvel], axis=1)
    elif target_mode == "sensor":
        y = trial.sensor
    elif target_mode == "ctrl":
        y = trial.ctrl
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    if lag < 0:
        raise ValueError("Negative lags are acausal and disabled for online training.")
    return y[lag:] if lag else y


def align_features_and_qpos(trial: Trial, features: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Align a precomputed feature matrix with qpos for online-compatible lags."""
    if lag < 0:
        raise ValueError("Negative lags are acausal and disabled for online evaluation.")
    if lag:
        return features[:-lag], trial.qpos[lag:]
    return features, trial.qpos


def build_matrix(
    trials: Iterable[Trial],
    dataset: Dataset,
    cfg: FeatureConfig,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack many trials into one X matrix and one target matrix for linear models."""
    channels = resolve_channels(dataset, cfg)
    xs, ys = [], []
    for trial in trials:
        x = build_features_for_trial(trial, channels, cfg)
        if cfg.lag:
            x = x[:-cfg.lag]
        y = aligned_target(trial, target_mode, cfg.lag)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Misaligned trial {trial.gesture}/{trial.repeat}: X={x.shape} Y={y.shape}")
        xs.append(x)
        ys.append(y.astype(np.float32, copy=False))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def trial_lengths(trials: Iterable[Trial]) -> list[int]:
    """Return trial lengths for simple exploration plots/tables."""
    return [t.n_time for t in trials]
