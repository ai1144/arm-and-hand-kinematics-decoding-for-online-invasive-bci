"""V1 presentation figure: all 32 qpos for one held-out trial of each object.

Three traces per joint: the reference posture, the V1 GRU with object ID and the
V1 Transformer without object ID. Both models are evaluated on the same
held-out repeat-holdout trial. Decoded postures come from the saved V1 replay
exports (32 posture values written in simulator control order, inverted here
with the exact one-to-one mapping). The reference posture is the true qpos of
the same trial, taken from the saved test arrays (identical data and split in
both runs).
"""
from pathlib import Path
from types import SimpleNamespace
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .results import OBJECTS, RUNS, candidate
from .simulation import viewer_module

COLORS = ['#222222', '#2ca58d', '#7b6bb7']          # reference, GRU, Transformer (thesis palette)
LABELS = ['Reference posture', 'GRU with object ID', 'Transformer without object ID']
LIMB_COLORS = {'Shoulder': '#4e79a7', 'Elbow': '#76b7b2', 'Wrist': '#59a14f', 'Thumb': '#f28e2b',
               'Index': '#e15759', 'Middle': '#b07aa1', 'Ring': '#c9a227', 'Little': '#9c755f'}
PRETTY = {'small_ball': 'Small ball', 'big_ball': 'Large ball', 'small_v_cylinder': 'Small vertical cylinder',
          'big_v_cylinder': 'Large vertical cylinder', 'small_h_cylinder': 'Small horizontal cylinder',
          'big_h_cylinder': 'Large horizontal cylinder'}


def limb(name):
    """Limb group of a qpos coordinate name (used for label colours)."""
    first = name.split('_')[0]
    return {'T': 'Thumb', 'I': 'Index', 'M': 'Middle'}.get(first, first)


def joint_label(name):
    return name.replace('_', ' ').replace('T ', 'Thumb ').replace('I ', 'Index ').replace('M ', 'Middle ')


def reference_qpos(root, key):
    """True qpos of repeat-holdout test trials from the saved test arrays."""
    root = Path(root)
    code = root / 'run_v2/functions'
    sys.path.insert(0, str(code)); sys.path.insert(0, str(code / 'scripts'))
    from run_split_pca_model_suite import make_split
    index = pd.read_csv(root / 'run_v2/results/combined_trial_index.csv')
    test = make_split([SimpleNamespace(**r) for r in index.to_dict('records')], 'repeat_holdout', 20260908)[2]
    source = sorted((root / 'run_v2/results/predictions').glob('repeat_holdout__*.npz'))[0]
    out, offset = {}, 0
    with np.load(source, allow_pickle=False) as arrays:
        assert np.array_equal(arrays['lengths'], [t.n_time for t in test])
        for trial, n in zip(test, arrays['lengths']):
            out[(trial.session, int(trial.gesture), int(trial.repeat))] = arrays['qpos_true'][offset:offset + int(n)]
            offset += int(n)
    return out[key]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def select_trials(root, run):
    """One held-out test trial per object with reference, GRU (object ID) and Transformer (no object ID) posture."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'run_v1/functions'))
    from decoder_core.mujoco_metadata import parse_mujoco_metadata
    meta = parse_mujoco_metadata(root / 'run_v1/functions/new_info/mujoco-model/mujoco_arm_1.1.xml')
    permutation = np.asarray(meta.qpos_to_ctrl_index, dtype=int)       # simulator order -> posture order
    assert sorted(permutation.tolist()) == list(range(32))
    gru = candidate(run, True, family='gru')
    transformer = candidate(run, False, family='transformer')
    viewer = viewer_module(root)
    replays = root / RUNS[run['run']] / 'results/replays'
    first = viewer.load_compare_pairs(replays / gru.experiment_id)
    second = viewer.load_compare_pairs(replays / transformer.experiment_id)
    key = lambda p: (p.metadata['session'], p.gesture, p.repeat)
    other = {key(p): p for p in second}
    selected, records = [], []
    for obj in OBJECTS:
        pair = next(p for p in sorted(first, key=key) if p.metadata['object_name'] == obj and key(p) in other)
        peer = other[key(pair)]
        ref = reference_qpos(root, key(pair))
        dec_gru = viewer.load_ctrl(pair.decoded_path, 32)[permutation].T
        dec_tf = viewer.load_ctrl(peer.decoded_path, 32)[permutation].T
        n = min(len(ref), len(dec_gru), len(dec_tf)); arrays = [ref[:n], dec_gru[:n], dec_tf[:n]]
        assert all(np.isfinite(a).all() for a in arrays)
        selected.append((obj, pair, arrays))
        records.append(dict(object=obj, session=key(pair)[0], gesture=pair.gesture, repeat=pair.repeat, samples=n,
                            gru_rmse_rad=rmse(arrays[0], arrays[1]), transformer_rmse_rad=rmse(arrays[0], arrays[2]),
                            gru_experiment=gru.experiment_id, transformer_experiment=transformer.experiment_id))
    return list(meta.qpos_names), selected, records


def overview_figure(names, selected, records, portrait=False):
    """Six objects side by side; every row is one of the 32 joints, scaled to its reference range."""
    handles = [plt.Line2D([], [], color=c, lw=2.4) for c in COLORS]
    figsize, label_size, title_size = ((8.3, 11), 7, 10) if portrait else ((16, 7.3), 7.4, 12.5)
    fig, axes = plt.subplots(1, 6, figsize=figsize, sharey=True)
    fig.subplots_adjust(left=.17 if portrait else .105, right=.995, bottom=.05, top=.87 if portrait else .845, wspace=.08)
    for ax, (obj, pair, arrays), rec in zip(axes, selected, records):
        ref = arrays[0]; center = ref.mean(axis=0); scale = np.maximum(np.ptp(ref, axis=0), .05)
        time = np.arange(len(ref)) / 100
        for i in range(32):
            for a, c, lw in zip(arrays, COLORS, [1.4, 1.0, 1.0]):
                ax.plot(time, 31 - i + (a[:, i] - center[i]) / scale[i] * .72, color=c, lw=lw, alpha=.95)
        for boundary in [i for i in range(1, 32) if limb(names[i]) != limb(names[i - 1])]:
            ax.axhline(31.5 - boundary, color='#c8ccd0', lw=.6)
        ax.set_title(PRETTY[obj].replace(' cylinder', '\ncylinder').replace('Small ball', 'Small\nball').replace('Large ball', 'Large\nball'),
                     fontsize=title_size, color='#0065BD', pad=4)
        ax.text(.5, -.02, f"RMSE\nGRU {rec['gru_rmse_rad']:.3f}\nTransf. {rec['transformer_rmse_rad']:.3f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=label_size, color='#333333')
        ax.set_xticks([]); ax.set_ylim(-.9, 32.1)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
    axes[0].set_yticks(range(32)); axes[0].set_yticklabels([joint_label(n) for n in reversed(names)], fontsize=label_size)
    for lab, n in zip(axes[0].get_yticklabels(), reversed(names)):
        lab.set_color(LIMB_COLORS.get(limb(n), '#333333'))
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.55, 1.0), ncol=3 if not portrait else 2,
               frameon=False, fontsize=9.5 if portrait else 13)
    reps = sorted({r['repeat'] + 1 for r in records})
    fig.text(.55, .93 if portrait else .925,
             f"One held-out trial per object (repetition {'/'.join(map(str, reps))} of 10, never used for training).\n"
             'Each row is one joint, scaled to its reference range; time runs left to right (about 5.5 s); RMSE in rad.',
             ha='center', fontsize=8 if portrait else 9.5, color='#555555')
    return fig


def detail_figure(names, obj, pair, arrays, rec, portrait=False):
    """All 32 joints of one trial in radians (8 x 4 small panels)."""
    handles = [plt.Line2D([], [], color=c, lw=2.4) for c in COLORS]
    fig, axes = plt.subplots(8, 4, figsize=(8.3, 11) if portrait else (16, 7.6), sharex=True)
    fig.subplots_adjust(left=.07, right=.99, bottom=.05, top=.88 if portrait else .85, hspace=.6 if portrait else .75, wspace=.3)
    time = np.arange(len(arrays[0])) / 100
    for i, ax in enumerate(axes.flat):
        for a, c, lw in zip(arrays, COLORS, [1.4, 1.0, 1.0]):
            ax.plot(time, a[:, i], color=c, lw=lw)
        ax.set_title(joint_label(names[i]), fontsize=8, pad=2, color=LIMB_COLORS.get(limb(names[i]), '#333333'))
        ax.tick_params(labelsize=6.5); ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
        if i // 4 == 7: ax.set_xlabel('Time from trial onset (s)', fontsize=7.5)
    fig.suptitle(f"{PRETTY[obj]}: all 32 posture coordinates (rad)", fontsize=14 if portrait else 17, y=.995, color='#0065BD')
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False, fontsize=9.5 if portrait else 11)
    fig.text(.5, .915 if portrait else .885, f"Held-out trial {pair.metadata['session']}, movement {pair.gesture}, repetition {pair.repeat + 1} of 10  |  "
             f"RMSE GRU {rec['gru_rmse_rad']:.3f} rad, Transformer {rec['transformer_rmse_rad']:.3f} rad",
             ha='center', fontsize=8.5 if portrait else 9.5, color='#555555')
    return fig


def generate(root, run, out, dpi=220):
    """Presentation version (16:9 PNGs and a multi-page PDF) for the V1 run."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    names, selected, records = select_trials(root, run)
    pd.DataFrame(records).to_csv(out / 'selected_test_trials.csv', index=False)
    pages = [overview_figure(names, selected, records)]
    pages[0].savefig(out / 'v1_all_qpos_six_objects.png', dpi=dpi, facecolor='white')
    for (obj, pair, arrays), rec in zip(selected, records):
        fig = detail_figure(names, obj, pair, arrays, rec)
        fig.savefig(out / f'v1_qpos_{obj}.png', dpi=dpi, facecolor='white')
        pages.append(fig)
    with PdfPages(out / 'v1_all_qpos_six_objects.pdf') as document:
        for fig in pages:
            document.savefig(fig); plt.close(fig)
    return pd.DataFrame(records)


def appendix_pages(root, run, out):
    """Portrait versions of the same figures for the thesis appendix (one PDF per page)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    names, selected, records = select_trials(root, run)
    fig = overview_figure(names, selected, records, portrait=True)
    fig.savefig(out / 'v1_appendix_qpos_overview.pdf'); fig.savefig(out / 'v1_appendix_qpos_overview.png', dpi=200); plt.close(fig)
    for (obj, pair, arrays), rec in zip(selected, records):
        fig = detail_figure(names, obj, pair, arrays, rec, portrait=True)
        fig.savefig(out / f'v1_appendix_qpos_{obj}.pdf'); fig.savefig(out / f'v1_appendix_qpos_{obj}.png', dpi=200); plt.close(fig)
    return pd.DataFrame(records)
