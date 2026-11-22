#!/usr/bin/env python3
"""MuJoCo replay viewer for actual-vs-decoded arm trajectories.

This script does not train models.  It is used after training, on the Mac (where this was built and tested) with
`mjpython`, to open exported `mujoco_compare/<experiment_id>/` folders and show
the ground-truth arm and decoded arm in the same MuJoCo scene.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PACKAGE_ROOT / "new_info" / "mujoco-model" / "mujoco_world_1.1.xml"
DEFAULT_GITLAB_DATA = PACKAGE_ROOT / "new_info" / "data"
DEFAULT_EXPORTED_CTRL = PACKAGE_ROOT / "artifacts" / "mujoco_ctrl"
DEFAULT_COMPARE_DIR = PACKAGE_ROOT / "artifacts" / "mujoco_compare"
DEFAULT_RUNTIME_MODEL_DIR = PACKAGE_ROOT / "artifacts" / "mujoco_runtime_model"
DEFAULT_COMPARE_MODEL_DIR = PACKAGE_ROOT / "artifacts" / "mujoco_compare_model"
OBJECT_NAMES = (
    "none",
    "small_ball",
    "big_ball",
    "small_v_cylinder",
    "big_v_cylinder",
    "small_h_cylinder",
    "big_h_cylinder",
)
START_SOUND = Path("/System/Library/Sounds/Tink.aiff")
END_SOUND = Path("/System/Library/Sounds/Pop.aiff")
BASE_ARM_BODIES = {
    "upperarm",
    "lowerarm",
    "wrist",
    "hand",
    "f1pp",
    "f1mp",
    "f1dp",
    "f2pp",
    "f2mp",
    "f2dp",
    "f3pp",
    "f3mp",
    "f3dp",
    "f4pp",
    "f4mp",
    "f4dp",
    "f5pp",
    "f5mp",
    "f5dp",
}
CTRL_DIM_TO_BODY_NAMES = {
    0: ["f1pp", "f1mp", "f1dp"],
    1: ["f1pp", "f1mp", "f1dp"],
    2: ["f1pp", "f1mp", "f1dp"],
    3: ["f1mp", "f1dp"],
    4: ["f1dp"],
    5: ["f2pp", "f2mp", "f2dp"],
    6: ["f2pp", "f2mp", "f2dp"],
    7: ["f2pp", "f2mp", "f2dp"],
    8: ["f2mp", "f2dp"],
    9: ["f2dp"],
    10: ["f3pp", "f3mp", "f3dp"],
    11: ["f3pp", "f3mp", "f3dp"],
    12: ["f3pp", "f3mp", "f3dp"],
    13: ["f3mp", "f3dp"],
    14: ["f3dp"],
    15: ["f4pp", "f4mp", "f4dp"],
    16: ["f4pp", "f4mp", "f4dp"],
    17: ["f4pp", "f4mp", "f4dp"],
    18: ["f4mp", "f4dp"],
    19: ["f4dp"],
    20: ["f5pp", "f5mp", "f5dp"],
    21: ["f5pp", "f5mp", "f5dp"],
    22: ["f5pp", "f5mp", "f5dp"],
    23: ["f5mp", "f5dp"],
    24: ["f5dp"],
    25: ["wrist", "hand", "f1pp", "f1mp", "f1dp", "f2pp", "f2mp", "f2dp", "f3pp", "f3mp", "f3dp", "f4pp", "f4mp", "f4dp", "f5pp", "f5mp", "f5dp"],
    26: ["wrist", "hand", "f1pp", "f1mp", "f1dp", "f2pp", "f2mp", "f2dp", "f3pp", "f3mp", "f3dp", "f4pp", "f4mp", "f4dp", "f5pp", "f5mp", "f5dp"],
    27: ["wrist", "hand", "f1pp", "f1mp", "f1dp", "f2pp", "f2mp", "f2dp", "f3pp", "f3mp", "f3dp", "f4pp", "f4mp", "f4dp", "f5pp", "f5mp", "f5dp"],
    28: ["lowerarm", "wrist", "hand"],
    29: ["upperarm", "lowerarm", "wrist", "hand"],
    30: ["upperarm", "lowerarm", "wrist", "hand"],
    31: ["upperarm", "lowerarm", "wrist", "hand"],
}


@dataclass(slots=True)
class ComparePair:
    gesture: int
    repeat: int
    actual_path: Path
    decoded_path: Path
    frames: int | None = None
    metadata: dict = field(default_factory=dict)


def import_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The Python package 'mujoco' is not installed in this environment.\n"
            "Install it with:\n\n"
            "    python3 -m pip install mujoco\n\n"
            "Then rerun this script."
        ) from exc
    return mujoco


def runtime_model_copy(path: Path, out_dir: Path = DEFAULT_RUNTIME_MODEL_DIR) -> Path:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(path.parent, out_dir)
    runtime_path = out_dir / path.name
    arm_path = out_dir / "mujoco_arm_1.1.xml"
    if arm_path.exists():
        arm_path.write_text(_soften_arm_geoms(arm_path.read_text()))
    text = runtime_path.read_text()
    text = re.sub(r'\s+apirate="[^"]*"', "", text)
    runtime_path.write_text(text)
    return runtime_path


def _soften_arm_geoms(arm_text: str) -> str:
    """Keep the original rectangular arm links in runtime visualization copies.

    Earlier exports converted the long upper/lower-arm boxes to capsules, but
    on macOS MuJoCo those capsule side faces could render almost black from the
    full-arm camera.  Keeping the original boxes is closer to the review
    reference video and makes the colored actual/decoded meshes easier to read.
    """
    return arm_text


def _balanced_tag_fragment(text: str, start: int, tag: str) -> str:
    open_re = re.compile(rf"<{tag}\b")
    close_re = re.compile(rf"</{tag}>")
    pos = start
    depth = 0
    while True:
        next_open = open_re.search(text, pos)
        next_close = close_re.search(text, pos)
        if next_close is None:
            raise ValueError(f"Could not find closing </{tag}> after offset {start}.")
        if next_open is not None and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            pos = next_close.end()
            if depth == 0:
                return text[start:pos]


def _extract_arm_body(arm_text: str) -> str:
    start = arm_text.find('<body childclass="arm"')
    if start < 0:
        raise ValueError("Could not find the root arm body in mujoco_arm_1.1.xml.")
    return _balanced_tag_fragment(arm_text, start, "body")


def _extract_actuators(arm_text: str) -> str:
    match = re.search(r"<actuator>\s*(.*?)\s*</actuator>", arm_text, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find the actuator block in mujoco_arm_1.1.xml.")
    return match.group(1)


def _set_geom_material(fragment: str, material_name: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "material=" in tag:
            return re.sub(r'material="[^"]+"', f'material="{material_name}"', tag)
        if tag.endswith("/>"):
            return tag[:-2] + f' material="{material_name}"/>'
        return tag[:-1] + f' material="{material_name}">'

    return re.sub(r"<geom\b[^>]*>", repl, fragment)


def _assign_material_to_root_arm(arm_text: str, material_name: str) -> str:
    start = arm_text.find('<body childclass="arm"')
    if start < 0:
        return arm_text
    original = _balanced_tag_fragment(arm_text, start, "body")
    updated = _set_geom_material(original, material_name)
    return arm_text[:start] + updated + arm_text[start + len(original) :]


def _prefix_name_attrs(fragment: str, prefix: str) -> str:
    return re.sub(r'\bname="([^"]+)"', lambda m: f'name="{prefix}{m.group(1)}"', fragment)


def _prefix_joint_refs(fragment: str, prefix: str) -> str:
    return re.sub(r'\bjoint="([^"]+)"', lambda m: f'joint="{prefix}{m.group(1)}"', fragment)


def _set_root_arm_position(fragment: str, y_offset_m: float) -> str:
    match = re.search(r'(<body\s+childclass="arm"[^>]*\bpos=")([^"]+)(")', fragment)
    if not match:
        return fragment
    parts = [float(v) for v in match.group(2).split()]
    if len(parts) != 3:
        return fragment
    parts[1] += float(y_offset_m)
    replacement = f'{match.group(1)}{parts[0]:.6g} {parts[1]:.6g} {parts[2]:.6g}{match.group(3)}'
    return fragment[: match.start()] + replacement + fragment[match.end() :]


def _tint_decoded_geoms(fragment: str, rgba: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "rgba=" in tag:
            tag = re.sub(r'rgba="[^"]+"', f'rgba="{rgba}"', tag)
        elif tag.endswith("/>"):
            tag = tag[:-2] + f' rgba="{rgba}"/>'
        else:
            tag = tag[:-1] + f' rgba="{rgba}">'
        return tag

    return re.sub(r"<geom\b[^>]*>", repl, fragment)


def _viewer_target_xml(target: dict | None) -> str:
    if not target:
        return ""
    geom_type = str(target.get("type", "sphere"))
    if geom_type not in {"sphere", "box", "capsule", "cylinder"}:
        geom_type = "sphere"

    def triple(name: str, default: str) -> str:
        value = target.get(name)
        if isinstance(value, (list, tuple)) and value:
            return " ".join(f"{float(v):.6g}" for v in value)
        return default

    pos = triple("pos", "0.035 -0.025 0.078")
    size = triple("size", "0.022")
    rgba = triple("rgba", "0.9 0.9 0.9 0.85")
    euler = triple("euler", "0 0 0")
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(target.get("name", "synthetic_target")))
    return (
        f'        <body name="{name}" pos="{pos}" euler="{euler}">\n'
        f'            <geom type="{geom_type}" size="{size}" rgba="{rgba}" '
        'contype="0" conaffinity="0" group="4"/>\n'
        "        </body>\n"
    )


def compare_model_copy(
    path: Path,
    out_dir: Path = DEFAULT_COMPARE_MODEL_DIR,
    *,
    separation_m: float = 0.025,
    decoded_rgba: str = "0.1 0.75 1.0 0.62",
    viewer_target: dict | None = None,
) -> Path:
    """Build a temporary MJCF with a second arm driven by decoded controls.

    The source GitLab world includes one arm with 32 position actuators.  For
    visual comparison we keep that arm as the actual trajectory, add a second
    prefixed copy of the arm body, and add 32 more prefixed actuators.  The
    resulting model therefore has 64 controls: first actual, then decoded.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(path.parent, out_dir)

    arm_path = out_dir / "mujoco_arm_1.1.xml"
    world_path = out_dir / "mujoco_world_compare.xml"
    arm_text = _soften_arm_geoms(arm_path.read_text())
    arm_text = _assign_material_to_root_arm(arm_text, "actual_arm")
    arm_path.write_text(arm_text)
    duplicate_body = _extract_arm_body(arm_text)
    duplicate_body = _prefix_name_attrs(duplicate_body, "decoded_")
    duplicate_body = _set_root_arm_position(duplicate_body, separation_m)
    duplicate_body = _set_geom_material(duplicate_body, "decoded_arm")
    duplicate_body = _tint_decoded_geoms(duplicate_body, decoded_rgba)

    duplicate_actuators = _extract_actuators(arm_text)
    duplicate_actuators = _prefix_name_attrs(duplicate_actuators, "decoded_")
    duplicate_actuators = _prefix_joint_refs(duplicate_actuators, "decoded_")
    target_xml = _viewer_target_xml(viewer_target)

    world_text = f"""<mujoco>
    <compiler coordinate="local" angle="radian"/>
    <option>
        <flag clampctrl="disable" contact="enable" gravity="disable"/>
    </option>
    <include file="mujoco_arm_1.1.xml"/>
    <include file="./Turntable/Karussell-main.xml"/>

    <default>
        <default class="button">
            <geom contype="0" conaffinity="0"/>
            <joint damping="0.1" stiffness="10" armature="0"/>
        </default>
        <default class="handle">
            <geom contype="0" conaffinity="0"/>
        </default>
        <default class="handrest">
            <geom contype="0" conaffinity="0"/>
            <joint damping="2" stiffness="100" armature="0"/>
        </default>
    </default>

    <visual>
        <scale jointwidth="0.05" jointlength="0.05" contactwidth="0.1" contactheight="0.04" forcewidth="0.01"/>
        <rgba contactpoint="1 1 0 1"/>
        <map znear="0.05"/>
        <headlight ambient="0 0 0" diffuse="0.8 0.8 0.8" specular="0 0 0"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.9" rgb2="0.5 0.5 0.5"
                 width="256" height="256"/>
        <material name="actual_arm" emission="0.25" specular="0.02" shininess="0.05" rgba="1.0 0.38 0.0 1"/>
        <material name="decoded_arm" emission="0.9" specular="0.0" shininess="0.0" rgba="0.0 0.82 1.0 1"/>
        <material name="pm" specular="0" shininess="0" rgba="0 0 0 1"/>
    </asset>

    <worldbody>
        <body name="or" pos="0 0 0">
            <geom type="box" size="0.02 0.001 0.001" pos="0.02 0 0" contype="0" conaffinity="0" group="3" rgba="1 0 0 1"/>
            <geom type="box" size="0.001 0.02 0.001" pos="0 0.02 0" contype="0" conaffinity="0" group="3" rgba="0 0.5 0 1"/>
            <geom type="box" size="0.001 0.001 0.02" pos="0 0 0.02" contype="0" conaffinity="0" group="3" rgba="0 0 0.5 1"/>
        </body>
        <body pos="-0.11 -0.17 0.28" euler="-0.6 0.85 0" mocap="true">
            <light directional="false" cutoff="45" exponent="20" attenuation="0.3 0 0" pos="0 0 0" dir="1 0 0" diffuse="0.82 0.91 1.00" ambient="0 0 0" specular="0 0 0" castshadow="false"/>
        </body>
        <body pos="-0.200 -0.150 0.240" euler="0 0 0" mocap="true">
            <camera euler="0 -1.5708 -1.5708"/>
        </body>
        <body pos="0 0 0" euler="0 0 0" mocap="true">
            <geom type="sphere" size="5e-4" rgba="0.5 0.0 0.5 1.0" group="3" contype="0" conaffinity="0"/>
        </body>
{target_xml}
{duplicate_body}
    </worldbody>

    <actuator>
{duplicate_actuators}
    </actuator>
</mujoco>
"""
    world_path.write_text(world_text)
    return world_path


def load_model(mujoco, path: Path):
    try:
        return mujoco.MjModel.from_xml_path(str(path)), path
    except ValueError as exc:
        if "apirate" not in str(exc):
            raise
        runtime_path = runtime_model_copy(path)
        return mujoco.MjModel.from_xml_path(str(runtime_path)), runtime_path


def load_compare_model(mujoco, path: Path, separation_m: float, viewer_target: dict | None = None):
    runtime_path = compare_model_copy(path, separation_m=separation_m, viewer_target=viewer_target)
    return mujoco.MjModel.from_xml_path(str(runtime_path)), runtime_path


def sample_path(data_dir: Path, object_name: str, trajectory: int) -> Path:
    return data_dir / f"{object_name}_{trajectory}.npy"


def exported_path(export_dir: Path, gesture: int) -> Path:
    return export_dir / f"gesture_{gesture:02d}_ctrl.npy"


def compare_paths(compare_dir: Path, gesture: int, repeat: int) -> tuple[Path, Path]:
    return (
        compare_dir / f"gesture_{gesture:02d}_repeat_{repeat:02d}_actual_ctrl.npy",
        compare_dir / f"gesture_{gesture:02d}_repeat_{repeat:02d}_decoded_ctrl.npy",
    )


def resolve_compare_export_path(path_value: str | Path, compare_dir: Path) -> Path:
    """Resolve exported .npy paths after a visualization package is copied.

    Some older transformer summaries stored absolute paths from the original
    workstation.  For review/share packages, the .npy files live next to the
    copied summary, so the filename fallback is the most portable option.
    """
    path = Path(path_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, PACKAGE_ROOT / path, compare_dir / path])
    candidates.append(compare_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_compare_pairs(compare_dir: Path, repeat: int | None = None) -> list[ComparePair]:
    summary_path = compare_dir / "summary.json"
    pairs: list[ComparePair] = []
    seen_paths: set[tuple[Path, Path]] = set()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for row in summary.get("exports", []):
            row_repeat = int(row["repeat"])
            if repeat is not None and row_repeat != repeat:
                continue
            actual_path = resolve_compare_export_path(row["actual"], compare_dir)
            decoded_path = resolve_compare_export_path(row["decoded"], compare_dir)
            key = (actual_path.resolve(), decoded_path.resolve())
            if key in seen_paths:
                continue
            if actual_path.exists() and decoded_path.exists():
                seen_paths.add(key)
                pairs.append(
                    ComparePair(
                        gesture=int(row["gesture"]),
                        repeat=row_repeat,
                        actual_path=actual_path,
                        decoded_path=decoded_path,
                        frames=int(row["frames"]) if "frames" in row else None,
                        metadata=dict(row),
                    )
                )
    else:
        pattern = re.compile(r"gesture_(\d+)_repeat_(\d+)_actual_ctrl\.npy$")
        for actual_path in sorted(compare_dir.glob("gesture_*_repeat_*_actual_ctrl.npy")):
            match = pattern.match(actual_path.name)
            if not match:
                continue
            gesture = int(match.group(1))
            row_repeat = int(match.group(2))
            if repeat is not None and row_repeat != repeat:
                continue
            decoded_path = compare_dir / f"gesture_{gesture:02d}_repeat_{row_repeat:02d}_decoded_ctrl.npy"
            if decoded_path.exists():
                pairs.append(ComparePair(gesture, row_repeat, actual_path, decoded_path))
    pairs.sort(key=lambda p: (p.gesture, p.repeat))
    if not pairs:
        raise FileNotFoundError(f"No actual/decoded comparison pairs found in {compare_dir}")
    return pairs


def load_compare_viewer_target(compare_dir: Path) -> dict | None:
    summary_path = compare_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    target = summary.get("viewer_target")
    return target if isinstance(target, dict) else None


def load_ctrl(path: Path, nu: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".csv":
        rows = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ctrl_cols = [c for c in reader.fieldnames or [] if c.startswith("ctrl_")]
            ctrl_cols.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
            for row in reader:
                rows.append([float(row[c]) for c in ctrl_cols])
        arr = np.asarray(rows, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported trajectory file: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D ctrl trajectory, got shape {arr.shape} from {path}")
    if arr.shape[0] == nu:
        return arr
    if arr.shape[1] == nu:
        return arr.T
    raise ValueError(f"Expected one axis to equal model.nu={nu}, got shape {arr.shape} from {path}")


def configure_camera(viewer, model, camera_view: str = "arm") -> None:
    if camera_view == "hand":
        viewer.cam.azimuth = -28.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[:] = [0.033, -0.032, 0.082]
        viewer.cam.distance = 0.16
        model.vis.global_.fovy = 58
    elif camera_view == "full32":
        viewer.cam.azimuth = -22.0
        viewer.cam.elevation = -32.0
        viewer.cam.lookat[:] = [-0.015, -0.020, 0.105]
        viewer.cam.distance = 0.24
        model.vis.global_.fovy = 68
    else:
        viewer.cam.azimuth = -18.33
        viewer.cam.elevation = -34.6
        viewer.cam.lookat[:] = [-0.10647625, 0.07342901, 0.22164613]
        viewer.cam.distance = 0.36
        model.vis.global_.fovy = 80
    viewer.fixedcamid = -1
    viewer.trackbodyid = -1
    viewer.type = 0
    if hasattr(viewer, "orthographic"):
        viewer.orthographic = 0
    if hasattr(viewer, "othographic"):
        viewer.othographic = 0


def configure_object_groups(mujoco, model, object_name: str | None) -> None:
    for name in OBJECT_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name} geom")
        if geom_id >= 0:
            model.geom_group[geom_id] = 10

    if object_name and object_name != "none":
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_name} geom")
        if geom_id < 0:
            raise ValueError(f"Object geom not found in MuJoCo model: {object_name!r}")
        model.geom_group[geom_id] = 4


def configure_visibility(mujoco, model, viewer, object_name: str | None, show_sites: bool) -> None:
    viewer.opt.geomgroup[:] = 0
    viewer.opt.geomgroup[0] = 1  # world/helpers
    viewer.opt.geomgroup[1] = 1  # arm and hand
    viewer.opt.geomgroup[3] = 1  # axes / fingertip helper geoms if present

    configure_object_groups(mujoco, model, object_name)
    viewer.opt.geomgroup[4] = 1

    viewer.opt.sitegroup[:] = 1 if show_sites else 0


def configure_compare_colors(mujoco, model) -> None:
    actual_rgba = np.asarray([1.0, 0.38, 0.0, 1.0], dtype=np.float32)
    decoded_rgba = np.asarray([0.0, 0.82, 1.0, 1.0], dtype=np.float32)
    actual_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "actual_arm")
    decoded_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "decoded_arm")
    for mat_id, rgba in [(actual_mat, actual_rgba), (decoded_mat, decoded_rgba)]:
        if mat_id >= 0:
            model.mat_rgba[mat_id] = rgba
            model.mat_emission[mat_id] = 0.25
            model.mat_specular[mat_id] = 0.02
            model.mat_shininess[mat_id] = 0.05
    model.vis.headlight.ambient[:] = [1.0, 1.0, 1.0]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    if model.nlight:
        model.light_ambient[:, :] = [1.0, 1.0, 1.0]
        model.light_diffuse[:, :] = [0.75, 0.75, 0.75]
        model.light_castshadow[:] = 0
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        is_decoded = body_name.startswith("decoded_")
        base_body = body_name.removeprefix("decoded_")
        if base_body not in BASE_ARM_BODIES:
            continue
        if is_decoded:
            model.geom_matid[geom_id] = decoded_mat if decoded_mat >= 0 else -1
            model.geom_rgba[geom_id] = decoded_rgba
        else:
            model.geom_matid[geom_id] = actual_mat if actual_mat >= 0 else -1
            model.geom_rgba[geom_id] = actual_rgba


def play_sound_cue(kind: str, enabled: bool) -> None:
    if not enabled:
        return
    label = "please start attempt" if kind == "start" else "attempt ended"
    sound_path = START_SOUND if kind == "start" else END_SOUND
    print(f"Sound cue: {label}", flush=True)
    afplay = shutil.which("afplay")
    if afplay and sound_path.exists():
        try:
            subprocess.Popen([afplay, str(sound_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    sys.stdout.write("\a")
    sys.stdout.flush()


def hide_upperarm_geoms(mujoco, model) -> None:
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.removeprefix("decoded_") == "upperarm":
            model.geom_group[geom_id] = 9


def hide_decoded_geoms(mujoco, model) -> None:
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("decoded_"):
            model.geom_group[geom_id] = 9


def active_dims_for_pair(pair: ComparePair) -> list[int]:
    dims = pair.metadata.get("active_dims", []) if pair.metadata else []
    if isinstance(dims, str):
        return [int(x) for x in dims.replace(",", " ").split() if x.strip().lstrip("-").isdigit()]
    if isinstance(dims, (list, tuple)):
        out = []
        for value in dims:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                pass
        return out
    return []


def apply_active_dof_colors(mujoco, model, pair: ComparePair, ctrl: np.ndarray, t: int, enabled: bool) -> None:
    if not enabled:
        return
    dims = [d for d in active_dims_for_pair(pair) if 0 <= d < 32]
    if not dims:
        return
    prev_t = max(0, int(t) - 1)
    t = int(np.clip(t, 0, ctrl.shape[1] - 1))
    speed = float(np.max(np.abs(ctrl[dims, t] - ctrl[dims, prev_t]))) if ctrl.shape[1] else 0.0
    cue_rgba = np.asarray([0.0, 0.88, 0.18, 1.0] if speed > 1e-4 else [1.0, 0.05, 0.03, 1.0], dtype=np.float32)
    bodies = set()
    for dim in dims:
        bodies.update(CTRL_DIM_TO_BODY_NAMES.get(dim, []))
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("decoded_"):
            continue
        if body_name in bodies:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = cue_rgba


def _pair_label(pair: ComparePair) -> tuple[str, str]:
    meta = pair.metadata or {}
    parts = []
    session = meta.get("session") or meta.get("session_id")
    if session is not None:
        parts.append(f"session {session}")
    condition = meta.get("condition")
    if condition is not None:
        parts.append(str(condition))
    handle = meta.get("handle_id")
    if handle is not None:
        parts.append(f"handle {handle}")
    obj = meta.get("object_name")
    if obj:
        parts.append(str(obj))
    trial_idx = meta.get("trial_idx")
    if trial_idx is not None:
        parts.append(f"trial {trial_idx}")

    if parts:
        title = f"Example {pair.gesture:02d}  |  " + "  |  ".join(parts)
        detail_prefix = (
            "Example number is the exported viewer index, not necessarily an experimental gesture class."
        )
    else:
        title = f"Gesture {pair.gesture:02d} repeat {pair.repeat:02d}"
        detail_prefix = "Legacy export: gesture number comes from the original exported file name."
    return title, detail_prefix


def set_compare_overlay(mujoco, viewer, pair: ComparePair, compare_dir: Path, paused: bool) -> None:
    status = "paused" if paused else "playing"
    label, detail_prefix = _pair_label(pair)
    title = f"{label}  |  {status}"
    detail = (
        f"{detail_prefix}  |  orange=actual  cyan=decoded  |  "
        f"green=moving cue red=stop cue when enabled  |  source={compare_dir.name}  |  "
        "N/P next/prev, R restart, Space pause"
    )
    try:
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                title,
                detail,
            )
        )
    except AttributeError:
        pass


def replay(model, data, mujoco, viewer, ctrl: np.ndarray, factor: int, realtime: bool, loop: bool) -> None:
    t = 0
    dt = float(model.opt.timestep)
    while viewer.is_running():
        current = data.ctrl.copy()
        target = ctrl[:, t]
        for k in range(max(1, factor)):
            alpha = k / max(1, factor)
            data.ctrl[:] = (1.0 - alpha) * current + alpha * target
            mujoco.mj_step(model, data)
            viewer.sync()
            if realtime:
                time.sleep(dt)
            if not viewer.is_running():
                return
        t += 1
        if t >= ctrl.shape[1]:
            if not loop:
                return
            t = 0


def _key_name(key: int) -> str:
    if 0 <= key < 256:
        return chr(key).lower()
    return ""


def replay_compare_loop(
    model,
    data,
    mujoco,
    viewer,
    pairs: list[ComparePair],
    start_index: int,
    compare_dir: Path,
    factor: int,
    slow_loop_factor: int,
    realtime: bool,
    loop: bool,
    state: dict[str, int | bool],
    trim_start_bins: int = 0,
    trim_end_bins: int = 0,
    sound_cues: bool = False,
    cue_active_dof_colors: bool = False,
) -> None:
    index = start_index
    t = 0
    completed_once = False
    dt = float(model.opt.timestep)

    def load_current() -> tuple[np.ndarray, ComparePair]:
        pair = pairs[index]
        actual = load_ctrl(pair.actual_path, 32)
        decoded = load_ctrl(pair.decoded_path, 32)
        n = min(actual.shape[1], decoded.shape[1])
        start = max(0, int(trim_start_bins))
        stop = n - max(0, int(trim_end_bins))
        if stop <= start + 1:
            start, stop = 0, n
        ctrl = np.vstack([actual[:, start:stop], decoded[:, start:stop]]).astype(np.float32, copy=False)
        if model.nu != ctrl.shape[0]:
            raise ValueError(f"Comparison model must have model.nu={ctrl.shape[0]}, got {model.nu}.")
        print(
            f"{_pair_label(pair)[0]}: orange=actual, cyan=decoded, frames={ctrl.shape[1]} "
            f"(trimmed {start} start, {n - stop} end from original {n}).",
            flush=True,
        )
        return ctrl, pair

    ctrl, current_pair = load_current()
    data.ctrl[:] = ctrl[:, 0]
    mujoco.mj_forward(model, data)
    configure_compare_colors(mujoco, model)
    apply_active_dof_colors(mujoco, model, current_pair, ctrl, t, cue_active_dof_colors)
    set_compare_overlay(mujoco, viewer, current_pair, compare_dir, bool(state.get("paused", False)))
    play_sound_cue("start", sound_cues)
    while viewer.is_running():
        delta = int(state.get("delta", 0))
        if delta:
            state["delta"] = 0
            index = (index + delta) % len(pairs)
            ctrl, current_pair = load_current()
            t = 0
            configure_compare_colors(mujoco, model)
            apply_active_dof_colors(mujoco, model, current_pair, ctrl, t, cue_active_dof_colors)
            set_compare_overlay(mujoco, viewer, current_pair, compare_dir, bool(state.get("paused", False)))
            completed_once = False
            data.ctrl[:] = ctrl[:, 0]
            mujoco.mj_forward(model, data)
            play_sound_cue("start", sound_cues)
        if bool(state.get("reset", False)):
            state["reset"] = False
            t = 0
            completed_once = False
            configure_compare_colors(mujoco, model)
            apply_active_dof_colors(mujoco, model, current_pair, ctrl, t, cue_active_dof_colors)
            set_compare_overlay(mujoco, viewer, current_pair, compare_dir, bool(state.get("paused", False)))
            data.ctrl[:] = ctrl[:, 0]
            mujoco.mj_forward(model, data)
            play_sound_cue("start", sound_cues)
        if bool(state.get("paused", False)):
            set_compare_overlay(mujoco, viewer, current_pair, compare_dir, True)
            viewer.sync()
            time.sleep(0.03)
            continue
        set_compare_overlay(mujoco, viewer, current_pair, compare_dir, False)
        configure_compare_colors(mujoco, model)
        apply_active_dof_colors(mujoco, model, current_pair, ctrl, t, cue_active_dof_colors)

        current = data.ctrl.copy()
        target = ctrl[:, t]
        current_factor = slow_loop_factor if completed_once else factor
        for k in range(max(1, current_factor)):
            alpha = k / max(1, current_factor)
            data.ctrl[:] = (1.0 - alpha) * current + alpha * target
            mujoco.mj_step(model, data)
            viewer.sync()
            if realtime:
                time.sleep(dt)
            if not viewer.is_running():
                return
        if t < ctrl.shape[1] - 1:
            t += 1
        elif loop:
            play_sound_cue("end", sound_cues)
            t = 0
            completed_once = True
            data.ctrl[:] = ctrl[:, 0]
            mujoco.mj_forward(model, data)
            play_sound_cue("start", sound_cues)
        elif not completed_once:
            completed_once = True
            play_sound_cue("end", sound_cues)
            print("End of gesture. Press N/P to switch, R to replay, Space to pause.", flush=True)


def save_snapshot(
    mujoco,
    model,
    data,
    ctrl: np.ndarray,
    path: Path,
    frame: int,
    width: int,
    height: int,
    camera_view: str = "arm",
) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Snapshot mode needs Pillow. Install it with:\n\n"
            "    python3 -m pip install pillow\n"
        ) from exc
    frame = int(np.clip(frame, 0, ctrl.shape[1] - 1))
    for t in range(frame + 1):
        data.ctrl[:] = ctrl[:, t]
        for _ in range(6):
            mujoco.mj_step(model, data)
    for _ in range(30):
        data.ctrl[:] = ctrl[:, frame]
        mujoco.mj_step(model, data)
    if hasattr(model.vis, "global_"):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if camera_view == "hand":
        camera.azimuth = -28.0
        camera.elevation = -30.0
        camera.lookat[:] = [0.033, -0.032, 0.082]
        camera.distance = 0.16
    elif camera_view == "full32":
        camera.azimuth = -22.0
        camera.elevation = -32.0
        camera.lookat[:] = [-0.015, -0.020, 0.105]
        camera.distance = 0.24
    else:
        camera.azimuth = -18.33
        camera.elevation = -34.6
    visible_ids = [i for i in range(model.ngeom) if model.geom_group[i] in {1, 3, 4}]
    if visible_ids and camera_view == "arm":
        positions = data.geom_xpos[visible_ids]
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        camera.lookat[:] = (lo + hi) * 0.5
        camera.distance = max(0.25, float(np.linalg.norm(hi - lo)) * 1.8)
    else:
        camera.lookat[:] = [-0.10647625, 0.07342901, 0.22164613]
        camera.distance = 0.36
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[0] = 1
    scene_option.geomgroup[1] = 1
    scene_option.geomgroup[3] = 1
    scene_option.geomgroup[4] = 1
    scene_option.sitegroup[:] = 0
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
    except (AttributeError, IndexError):
        pass
    pixels = renderer.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    renderer.close()


def save_compare_snapshot(
    mujoco,
    model,
    data,
    pair: ComparePair,
    path: Path,
    frame: int,
    width: int,
    height: int,
    camera_view: str = "arm",
) -> None:
    actual = load_ctrl(pair.actual_path, 32)
    decoded = load_ctrl(pair.decoded_path, 32)
    n = min(actual.shape[1], decoded.shape[1])
    ctrl = np.vstack([actual[:, :n], decoded[:, :n]]).astype(np.float32, copy=False)
    save_snapshot(mujoco, model, data, ctrl, path, frame, width, height, camera_view)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay MuJoCo robot hand ctrl trajectories.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--object-name", choices=OBJECT_NAMES, default="small_h_cylinder")
    parser.add_argument("--trajectory", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_GITLAB_DATA)
    parser.add_argument("--gesture", type=int, default=None, help="Replay artifacts/mujoco_ctrl/gesture_XX_ctrl.npy instead of a GitLab sample.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORTED_CTRL)
    parser.add_argument("--ctrl-file", type=Path, default=None, help="Explicit .npy or .csv ctrl trajectory.")
    parser.add_argument("--compare", action="store_true", help="Show actual and decoded trajectories as two arm meshes in one MuJoCo window.")
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--repeat", type=int, default=8, help="Repeat number to use for --compare exports.")
    parser.add_argument("--compare-separation", type=float, default=0.025, help="Meters to shift decoded mesh in +Y. Use 0 for exact overlay.")
    parser.add_argument("--hide-decoded", action="store_true", help="In compare mode, hide the decoded/cyan arm copy for a clean single-arm playlist.")
    parser.add_argument(
        "--show-upperarm",
        action="store_true",
        help="Compatibility flag; full shoulder/upper-arm links are shown by default in compare mode.",
    )
    parser.add_argument("--hide-upperarm", action="store_true", help="Hide the shoulder/upper-arm links for a closer hand-only view.")
    parser.add_argument("--trim-start-bins", type=int, default=0, help="Skip this many leading exported bins in compare playback.")
    parser.add_argument("--trim-end-bins", type=int, default=0, help="Skip this many trailing exported bins in compare playback.")
    parser.add_argument("--factor", type=int, default=60, help="Interpolation substeps per ctrl sample. Higher is slower.")
    parser.add_argument("--slow-loop-factor", type=int, default=600, help="Interpolation factor after first pass when looping.")
    parser.add_argument("--loop", action="store_true", help="Loop the trajectory after it finishes.")
    parser.add_argument("--realtime", action="store_true", help="Sleep for model.opt.timestep after each MuJoCo step.")
    parser.add_argument("--show-sites", action="store_true", help="Show MuJoCo site markers.")
    parser.add_argument("--sound-cues", action="store_true", help="Play a light start beep and end beep during compare playback.")
    parser.add_argument("--cue-active-dof-colors", action="store_true", help="For primitive cue playlists, color the active actual-arm DOF green while moving and red while stopped.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Render one PNG frame instead of opening the viewer.")
    parser.add_argument("--snapshot-frame", type=int, default=0)
    parser.add_argument("--snapshot-width", type=int, default=900)
    parser.add_argument("--snapshot-height", type=int, default=700)
    parser.add_argument(
        "--camera-view",
        choices=["arm", "hand", "full32"],
        default="arm",
        help="Use the legacy full-arm, close-up hand, or balanced full-32DOF camera.",
    )
    args = parser.parse_args()

    mujoco = import_mujoco()
    pairs: list[ComparePair] = []
    start_index = 0
    if args.compare:
        pairs = load_compare_pairs(args.compare_dir, repeat=args.repeat)
        if args.gesture is not None:
            matches = [i for i, pair in enumerate(pairs) if pair.gesture == args.gesture]
            if not matches:
                raise FileNotFoundError(f"No comparison export for gesture {args.gesture} repeat {args.repeat}.")
            start_index = matches[0]
        viewer_target = load_compare_viewer_target(args.compare_dir)
        model, loaded_model_path = load_compare_model(
            mujoco,
            args.model,
            args.compare_separation,
            viewer_target=viewer_target,
        )
        configure_compare_colors(mujoco, model)
        if args.hide_upperarm:
            hide_upperarm_geoms(mujoco, model)
        if args.hide_decoded:
            hide_decoded_geoms(mujoco, model)
    else:
        model, loaded_model_path = load_model(mujoco, args.model)
    data = mujoco.MjData(model)

    if args.compare:
        ctrl_path = args.compare_dir
    elif args.ctrl_file is not None:
        ctrl_path = args.ctrl_file
    elif args.gesture is not None:
        ctrl_path = exported_path(args.export_dir, args.gesture)
    else:
        ctrl_path = sample_path(args.data_dir, args.object_name, args.trajectory)

    print(f"Model: {args.model}")
    if loaded_model_path != args.model:
        print(f"Runtime-compatible model copy: {loaded_model_path}")
    print(f"Trajectory: {ctrl_path}")
    if args.compare:
        print(f"Comparison mode: model.nu={model.nu}  model.nq={model.nq}")
        print("Viewer controls: N/right=next gesture, P/left=previous gesture, R=replay, Space=pause.")
    else:
        ctrl = load_ctrl(ctrl_path, model.nu)
        print(f"ctrl shape: {ctrl.shape}  model.nu={model.nu}  model.nq={model.nq}")
    if args.snapshot is not None:
        configure_object_groups(mujoco, model, args.object_name)
        if args.compare:
            save_compare_snapshot(
                mujoco,
                model,
                data,
                pairs[start_index],
                args.snapshot,
                args.snapshot_frame,
                args.snapshot_width,
                args.snapshot_height,
                args.camera_view,
            )
        else:
            save_snapshot(
                mujoco,
                model,
                data,
                ctrl,
                args.snapshot,
                args.snapshot_frame,
                args.snapshot_width,
                args.snapshot_height,
                args.camera_view,
            )
        print(f"Snapshot: {args.snapshot}")
        return 0
    print("Close the MuJoCo viewer window to stop.")

    compare_state: dict[str, int | bool] = {"delta": 0, "reset": False, "paused": False}

    def key_callback(key: int) -> None:
        name = _key_name(key)
        if name in {"n", "]"} or key == 262:  # right arrow
            compare_state["delta"] = int(compare_state.get("delta", 0)) + 1
        elif name in {"p", "["} or key == 263:  # left arrow
            compare_state["delta"] = int(compare_state.get("delta", 0)) - 1
        elif name == "r":
            compare_state["reset"] = True
        elif name == " ":
            compare_state["paused"] = not bool(compare_state.get("paused", False))

    viewer_kwargs = {"key_callback": key_callback} if args.compare else {}
    with mujoco.viewer.launch_passive(model, data, **viewer_kwargs) as viewer:
        configure_camera(viewer, model, args.camera_view)
        configure_visibility(mujoco, model, viewer, args.object_name, args.show_sites)
        viewer.sync()
        if args.compare:
            replay_compare_loop(
                model,
                data,
                mujoco,
                viewer,
                pairs,
                start_index,
                args.compare_dir,
                args.factor,
                args.slow_loop_factor,
                args.realtime,
                args.loop,
                compare_state,
                args.trim_start_bins,
                args.trim_end_bins,
                args.sound_cues,
                args.cue_active_dof_colors,
            )
        else:
            replay(model, data, mujoco, viewer, ctrl, args.factor, args.realtime, loop=False)
            while args.loop and viewer.is_running():
                replay(model, data, mujoco, viewer, ctrl, args.slow_loop_factor, args.realtime, loop=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
