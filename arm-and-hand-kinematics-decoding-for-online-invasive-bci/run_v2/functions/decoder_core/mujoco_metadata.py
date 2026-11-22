from __future__ import annotations
"""Read MuJoCo model metadata needed by the decoder pipeline.

The training target is qpos, but visualization uses ctrl.  This parser records
joint/control names and the qpos-to-ctrl mapping so exported trajectories can be
shown correctly in MuJoCo.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUJOCO_XML = PACKAGE_ROOT / "new_info" / "mujoco-model" / "mujoco_arm_1.1.xml"
DEFAULT_TURNTABLE_XML = PACKAGE_ROOT / "new_info" / "mujoco-model" / "Turntable" / "Karussell-main.xml"


def _strip_comments(text: str) -> str:
    """Remove XML comments before using lightweight regex parsing."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _attrs(text: str) -> dict[str, str]:
    """Extract XML attributes from one tag fragment into a dictionary."""
    return dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', text))


def _block(text: str, tag: str) -> str:
    """Return the text inside a top-level XML block, or empty text if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    return match.group(1) if match else ""


def _range_to_float_pair(value: str | None) -> list[float] | None:
    """Parse MuJoCo ctrlrange text like '-1 1' into two floats."""
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 2:
        return None
    return [float(parts[0]), float(parts[1])]


@dataclass(slots=True)
class MuJoCoMetadata:
    """Joint/control/sensor metadata extracted from the MuJoCo XML files."""

    xml_path: str
    qpos_names: list[str]
    ctrl_names: list[str]
    sensor_names: list[str]
    ctrlrange: list[list[float] | None]
    actuator_type: list[str]
    ctrl_to_qpos_index: list[int | None]
    qpos_to_ctrl_index: list[int | None]
    turntable_objects: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata for JSON manifests and debugging."""
        return {
            "xml_path": self.xml_path,
            "nq": len(self.qpos_names),
            "nu": len(self.ctrl_names),
            "nsensor": len(self.sensor_names),
            "qpos_names": self.qpos_names,
            "ctrl_names": self.ctrl_names,
            "sensor_names": self.sensor_names,
            "ctrlrange": self.ctrlrange,
            "actuator_type": self.actuator_type,
            "ctrl_to_qpos_index": self.ctrl_to_qpos_index,
            "qpos_to_ctrl_index": self.qpos_to_ctrl_index,
            "turntable_objects": self.turntable_objects,
            "coordinate_note": (
                "MuJoCo qpos order follows joint declaration order. "
                "MuJoCo ctrl order follows actuator declaration order. "
                "These two 32-D orders are different for this model."
            ),
        }


def parse_turntable_objects(path: Path = DEFAULT_TURNTABLE_XML) -> list[str]:
    """Read object names from the turntable XML, if that file is available."""
    if not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    names = []
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name and name.endswith(" geom"):
            names.append(name.removesuffix(" geom"))
    return names


def parse_mujoco_metadata(path: Path = DEFAULT_MUJOCO_XML) -> MuJoCoMetadata | None:
    """Parse qpos names, actuator ctrl names, sensor names, and qpos/ctrl mapping."""
    if not path.exists():
        return None

    text = _strip_comments(path.read_text())
    default_arm = ""
    arm_match = re.search(r'<default\s+class="arm">(.*?)</default>', text, flags=re.S)
    if arm_match:
        default_arm = arm_match.group(1)
    default_position = None
    pos_match = re.search(r"<position\s+([^>]+)/?>", default_arm)
    if pos_match:
        default_position = _attrs(pos_match.group(1))

    qpos_names: list[str] = []
    for match in re.finditer(r"<joint\s+([^>]+)/?>", text):
        attrs = _attrs(match.group(1))
        name = attrs.get("name")
        if name:
            qpos_names.append(name)

    ctrl_names: list[str] = []
    actuator_type: list[str] = []
    ctrlrange: list[list[float] | None] = []
    for match in re.finditer(r"<(position|velocity|motor)\s+([^>]+)/?>", _block(text, "actuator")):
        tag = match.group(1)
        attrs = _attrs(match.group(2))
        name = attrs.get("name") or attrs.get("joint")
        if not name:
            continue
        ctrl_names.append(name)
        actuator_type.append(tag)
        inherited = default_position.get("ctrlrange") if default_position else None
        ctrlrange.append(_range_to_float_pair(attrs.get("ctrlrange") or inherited))

    sensor_names: list[str] = []
    for match in re.finditer(r"<(\w+)\s+([^>]+)/?>", _block(text, "sensor")):
        attrs = _attrs(match.group(2))
        name = attrs.get("name")
        if name:
            sensor_names.append(name)

    qpos_index = {name: i for i, name in enumerate(qpos_names)}
    ctrl_index = {name: i for i, name in enumerate(ctrl_names)}
    ctrl_to_qpos_index = [qpos_index.get(name) for name in ctrl_names]
    qpos_to_ctrl_index = [ctrl_index.get(name) for name in qpos_names]

    return MuJoCoMetadata(
        xml_path=str(path),
        qpos_names=qpos_names,
        ctrl_names=ctrl_names,
        sensor_names=sensor_names,
        ctrlrange=ctrlrange,
        actuator_type=actuator_type,
        ctrl_to_qpos_index=ctrl_to_qpos_index,
        qpos_to_ctrl_index=qpos_to_ctrl_index,
        turntable_objects=parse_turntable_objects(),
    )
