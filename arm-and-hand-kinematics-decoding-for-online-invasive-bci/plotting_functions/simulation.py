"""MuJoCo replay of held-out test trials: reference posture next to decoded posture.

The saved replay files contain, for each exported test trial, the reference and the
decoded 32-value posture written in the simulator's actuator order. Both are played
through the same MuJoCo arm-and-hand model; the camera looks at the hand from the
side so that the hand shape (opening, closing, wrist rotation) is visible.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .results import OBJECTS, OBJECT_LABEL, RUNS, RUN_TITLE, FAMILY_LABEL, candidate
from .style import save, setup

setup()
PHASES = [('Rest', .08), ('Reach', .3), ('Grasp', .45), ('Hold', .62), ('Return', .85)]
REFERENCE_RGBA, DECODED_RGBA = (1.0, .45, 0.0, 1.0), (0.0, .75, 1.0, 1.0)


def viewer_module(root):
    """Import the original viewer helpers shipped with the run code."""
    path = Path(root) / 'run_v2/functions/scripts/view_mujoco_hand.py'
    spec = importlib.util.spec_from_file_location('view_mujoco_hand', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def available_trials(root, r, object_id=True):
    """Held-out test trials with a saved replay for the run's candidate decoder."""
    v = viewer_module(root)
    row = candidate(r, object_id)
    pairs = v.load_compare_pairs(Path(root) / RUNS[r['run']] / 'results/replays' / row.experiment_id)
    table = pd.DataFrame([dict(session=p.metadata['session'], trial=p.gesture, repetition=p.repeat,
                               object=p.metadata['object_name']) for p in pairs])
    return row, pairs, table


class Replay:
    """Plays one reference/decoded pair through MuJoCo and renders side-view frames."""

    def __init__(self, root, pair):
        import mujoco
        self.mj, self.v = mujoco, viewer_module(root)
        xml = Path(root) / 'run_v2/functions/new_info/mujoco-model/mujoco_world_1.1.xml'
        workdir = Path(tempfile.mkdtemp(prefix='thesis_mujoco_'))       # temporary copy of the scene
        runtime = self.v.compare_model_copy(xml.resolve(), out_dir=workdir, separation_m=0.0)
        self.model = mujoco.MjModel.from_xml_path(str(runtime))
        self.v.configure_compare_colors(mujoco, self.model)
        self._shading()
        self.v.configure_object_groups(mujoco, self.model, 'none')      # the scene object is not placed at the grasp location
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1400, 1100
        reference = self.v.load_ctrl(pair.actual_path, 32)
        decoded = self.v.load_ctrl(pair.decoded_path, 32)
        n = min(reference.shape[1], decoded.shape[1])
        ctrl = np.vstack([reference[:, :n], decoded[:, :n]])
        data = mujoco.MjData(self.model)
        hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self.states, self.hand = [], []
        for t in range(n):                                            # simulate once, store every bin
            data.ctrl[:] = ctrl[:, t]
            for _ in range(6):
                mujoco.mj_step(self.model, data)
            self.states.append(data.qpos.copy())
            self.hand.append(data.xpos[hand].copy())
        self.n, self.data = n, data
        self.groups = self.model.geom_group.copy()

    def _shading(self):
        """Directional light and matte materials, so that the joints and segments are visible."""
        mj, m = self.mj, self.model
        m.vis.headlight.ambient[:] = [.35, .35, .35]
        m.vis.headlight.diffuse[:] = [.75, .75, .75]
        m.vis.headlight.specular[:] = [.25, .25, .25]
        for name, rgba in [('actual_arm', (1.0, .5, .05, 1.0)), ('decoded_arm', (.0, .62, .92, 1.0))]:
            mat = mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, name)
            if mat >= 0:
                m.mat_rgba[mat] = rgba
                m.mat_emission[mat] = 0.0
                m.mat_specular[mat] = .35
                m.mat_shininess[mat] = .4
        for g in range(m.ngeom):
            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
            if body.removeprefix('decoded_') in self.v.BASE_ARM_BODIES:
                m.geom_rgba[g] = (.0, .62, .92, 1.0) if body.startswith('decoded_') else (1.0, .5, .05, 1.0)

    def show(self, which):
        """Show only the reference arm, only the decoded arm, or both."""
        self.model.geom_group[:] = self.groups
        for g in range(self.model.ngeom):
            body = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or ''
            decoded = body.startswith('decoded_')
            if body.removeprefix('decoded_') in self.v.BASE_ARM_BODIES and which != 'both' and decoded != (which == 'decoded'):
                self.model.geom_group[g] = 9

    def frame(self, fraction, which='both', distance=.24):
        t = int(round(fraction * (self.n - 1)))
        self.show(which)
        self.data.qpos[:] = self.states[t]
        self.mj.mj_forward(self.model, self.data)
        cam = self.mj.MjvCamera()
        cam.type = self.mj.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 90, -15, distance
        cam.lookat[:] = self.hand[t]                                  # follow the reference hand
        option = self.mj.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = option.geomgroup[1] = 1
        option.sitegroup[:] = 0                                      # hide marker sites
        with self.mj.Renderer(self.model, height=700, width=900) as renderer:
            renderer.update_scene(self.data, camera=cam, scene_option=option)
            image = renderer.render().copy()
        return image, t / 100.0


def replay_panel(root, r, out, session=None, trial=None, repetition=None, object_id=True):
    """Reference (top) and decoded (bottom) hand at five moments of one held-out trial."""
    row, pairs, _ = available_trials(root, r, object_id)
    match = [p for p in pairs if (session is None or p.metadata['session'] == session)
             and (trial is None or p.gesture == trial) and (repetition is None or p.repeat == repetition)]
    if not match:
        raise ValueError('This trial has no saved replay; choose one from available_trials().')
    pair = match[0]
    sim = Replay(root, pair)
    fig, axes = plt.subplots(2, len(PHASES), figsize=(11, 4.7))
    fig.subplots_adjust(left=.06, right=.995, top=.8, bottom=.02, wspace=.03, hspace=.06)
    for col, (name, f) in enumerate(PHASES):
        for k, (which, label) in enumerate([('reference', 'Reference'), ('decoded', 'Decoded')]):
            image, seconds = sim.frame(f, which)
            axes[k, col].imshow(image)
            axes[k, col].set_xticks([]); axes[k, col].set_yticks([])
            for spine in axes[k, col].spines.values():
                spine.set_visible(True); spine.set_color('#bbbbbb')
            if k == 0:
                axes[k, col].set_title(name, fontsize=12)
        axes[0, 0].set_ylabel('Reference', fontsize=12, color='#e37222', fontweight='bold')
        axes[1, 0].set_ylabel('Decoded', fontsize=12, color='#0098c8', fontweight='bold')
    obj = OBJECT_LABEL[pair.metadata['object_name']].replace('\n', ' ').lower()
    fig.suptitle(f"{RUN_TITLE[r['run']]} — {FAMILY_LABEL[row.model_family]}, held-out trial ({obj})",
                 fontsize=12.5, fontweight='bold', y=.995)
    source = pd.DataFrame([dict(experiment_id=row.experiment_id, session=pair.metadata['session'], trial=pair.gesture,
                                repetition=pair.repeat, object=pair.metadata['object_name'], frames=sim.n)])
    return save(fig, out, f"{r['run']}_replay_side_view", source, dpi=220)


def six_objects(root, r, out):
    """Appendix: one held-out trial per object, reference (orange) and decoded (cyan) overlaid."""
    row, pairs, _ = available_trials(root, r, True)
    shown = [('Reach', .3), ('Grasp', .45), ('Return', .85)]
    fig, axes = plt.subplots(6, 3, figsize=(8, 13))
    fig.subplots_adjust(left=.13, right=.995, top=.95, bottom=.01, wspace=.03, hspace=.06)
    rows = []
    for i, obj in enumerate(OBJECTS):
        pair = next(p for p in pairs if p.metadata['object_name'] == obj)
        sim = Replay(root, pair)
        for j, (name, f) in enumerate(shown):
            image, seconds = sim.frame(f, 'both', distance=.26)
            axes[i, j].imshow(image)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(name, fontsize=13)
        axes[i, 0].set_ylabel(OBJECT_LABEL[obj], fontsize=11)
        rows.append(dict(object=obj, session=pair.metadata['session'], trial=pair.gesture, repetition=pair.repeat))
    fig.suptitle(f"{RUN_TITLE[r['run']]} — {FAMILY_LABEL[row.model_family]}: orange reference, cyan decoded",
                 fontsize=13, fontweight='bold')
    return save(fig, out, f"{r['run']}_replay_six_objects", pd.DataFrame(rows).assign(experiment_id=row.experiment_id), dpi=200)


def reference_strip(root, r, out, object_name='big_v_cylinder'):
    """Data chapter: the prescribed reference movement of one trial, seen from the side."""
    _, pairs, _ = available_trials(root, r, True)
    pair = next(p for p in pairs if p.metadata['object_name'] == object_name)
    sim = Replay(root, pair)
    fig, axes = plt.subplots(1, len(PHASES), figsize=(11, 2.6))
    fig.subplots_adjust(left=.005, right=.995, top=.84, bottom=.02, wspace=.03)
    for ax, (name, f) in zip(axes, PHASES):
        image, seconds = sim.frame(f, 'reference')
        ax.imshow(image)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=11)
    return save(fig, out, 'reference_movement_strip', dpi=220)


def replay_animation(root, r, out, session=None, trial=None, repetition=None, step=5):
    """Animated GIF: reference (left) and decoded (right) movement of one held-out trial, played in real time."""
    from PIL import Image as PILImage
    row, pairs, _ = available_trials(root, r, True)
    pair = next(p for p in pairs if (session is None or p.metadata['session'] == session)
                and (trial is None or p.gesture == trial) and (repetition is None or p.repeat == repetition))
    sim = Replay(root, pair)
    frames = []
    for t in range(0, sim.n, step):
        f = t / (sim.n - 1)
        left, seconds = sim.frame(f, 'reference', distance=.3)
        right, _ = sim.frame(f, 'decoded', distance=.3)
        image = PILImage.fromarray(np.hstack([left, right])).resize((900, 350))
        frames.append(image)
    path = Path(out) / f"{r['run']}_replay_animation.gif"
    Path(out).mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=step * 10, loop=0, optimize=True)
    return path


# Joint angles (rad) of a thumbs-up: fingers curled into a fist, thumb extended, palm facing sideways
THUMBS_UP = {'Shoulder_flex': .3, 'Shoulder_abd-add': .05, 'Shoulder_pro-sup': 0., 'Elbow': 1.6,
             'Wrist_pro-sup': 3.0, 'Wrist_rad_dev': -.3, 'Wrist_flex': 0.,
             'T_MCP_flexion': -.5, 'T_MCP_spread': -.45, 'T_MCP_rotation': .1, 'T_DIP': 0., 'T_PIP': 0.}   # straight thumb
for _finger in ['I', 'M', 'Ring', 'Little']:
    THUMBS_UP.update({f'{_finger}_MCP_flexion': 1.55, f'{_finger}_MCP_spread': 0., f'{_finger}_MCP_rotation': 0.,
                      f'{_finger}_PIP': 1.9, f'{_finger}_DIP': 1.3})


def thanks(root, out):
    """The arm-and-hand model giving a thumbs-up (last page of the thesis)."""
    import mujoco
    v = viewer_module(root)
    xml = (Path(root) / 'run_v2/functions/new_info/mujoco-model/mujoco_world_1.1.xml').resolve()
    model = mujoco.MjModel.from_xml_path(str(v.runtime_model_copy(xml, out_dir=Path(tempfile.mkdtemp(prefix='thesis_mujoco_')))))
    v.configure_object_groups(mujoco, model, 'none')
    model.vis.global_.offwidth, model.vis.global_.offheight = 1400, 1100
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:], model.vis.headlight.specular[:] = .35, .75, .25
    for g in range(model.ngeom):                                    # forearm and hand in one matte colour, rest hidden
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''
        visible = body in v.BASE_ARM_BODIES and body != 'upperarm'
        model.geom_matid[g] = -1
        model.geom_rgba[g] = (.0, .62, .92, 1.0) if visible else (0, 0, 0, 0)   # cyan of the decoded replays
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    data.qpos[:] = [THUMBS_UP[n] for n in names]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 160, 12, .2         # thumb side, curled fingers toward the viewer
    cam.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')]
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.geomgroup[:] = 0
    option.geomgroup[0] = option.geomgroup[1] = 1
    with mujoco.Renderer(model, height=900, width=900) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=option)
        image = renderer.render().copy()
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(image); ax.axis('off')
    return save(fig, out, 'thanks_arm')


def object_sprites(root, colors, size=360, azimuth=135, elevation=-30):
    """MuJoCo renders of the six task objects, one per colour, with a transparent background (RGBA images).

    `colors` maps a key (for example the block number) to a hex colour; the result is keyed by (object, key)."""
    import mujoco
    from matplotlib.colors import to_rgb
    v = viewer_module(root)
    xml = (Path(root) / 'run_v2/functions/new_info/mujoco-model/mujoco_world_1.1.xml').resolve()
    model = mujoco.MjModel.from_xml_path(str(v.runtime_model_copy(xml, out_dir=Path(tempfile.mkdtemp(prefix='thesis_mujoco_')))))
    model.vis.global_.offwidth = model.vis.global_.offheight = size
    model.vis.map.znear = 0.0002                                    # close-up renders of small objects
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:], model.vis.headlight.specular[:] = .45, .7, .3
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    body = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
    sprites = {}
    with mujoco.Renderer(model, size, size) as renderer:
        for name in OBJECTS:
            target = [g for g in range(model.ngeom) if body(g) == name][0]
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = data.geom_xpos[target]
            camera.azimuth, camera.elevation = azimuth, elevation
            camera.distance = 4.2 * float(np.max(model.geom_size[target]))
            for key, color in colors.items():
                model.geom_rgba[:] = 0                              # every other geom invisible
                model.geom_matid[target], model.geom_group[target] = -1, 0
                model.geom_rgba[target] = (*to_rgb(color), 1)
                renderer.disable_segmentation_rendering()
                renderer.update_scene(data, camera=camera, scene_option=option)
                image = renderer.render().copy()
                renderer.enable_segmentation_rendering()             # object mask = transparency
                renderer.update_scene(data, camera=camera, scene_option=option)
                mask = renderer.render()[..., 0] == target
                sprites[(name, key)] = np.dstack([image, mask.astype(np.uint8) * 255])
    return sprites
