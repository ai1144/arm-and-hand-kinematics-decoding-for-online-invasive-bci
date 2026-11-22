"""Print the trial durations of the nine recording blocks (read-only; nothing is written).
Run on the machine that holds the recordings:   python3 print_trial_durations.py
It compares, for every block, the recorded attempted-movement trials (trials-attempt.pkl), the resampled
trials used for training (trials-attempt-resampled.pkl) and the reference kinematics (all_attempt.pkl),
at 100 Hz (one bin = 10 ms). The printed values are stored in recorded_attempt_durations.csv and are
checked in Section 3.3 of the notebook.
"""
import pickle
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/vmshare')
BLOCKS = [f'{d}-block{b}' for d in ('20260630', '20260703', '20260707') for b in (1, 2, 3)]
HZ = 100.0


def load(path):
    try:
        import pandas as pd
        return pd.read_pickle(path)
    except Exception:
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def n_bins(x):
    """Length in time bins of one trial: rows of a kinematic table, or the time dimension of a neural array."""
    if hasattr(x, 'columns'):
        return len(x)
    x = np.asarray(x)
    return max(x.shape) if x.ndim == 2 else len(x)   # trials (500-750 bins) are longer than the channel dimension


def flatten(obj):
    """Trials as a flat list, whatever the nesting (list of trials or list of movements with repetitions)."""
    if isinstance(obj, dict):
        obj = list(obj.values())
    out = []
    for item in obj:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def describe(name, path):
    if not path.exists():
        print(f'  {name:32s} not found')
        return None, np.array([])
    obj = load(path)
    trials = flatten(obj)
    lengths = np.array([n_bins(t) for t in trials])
    print(f'  {name:32s} {len(trials):3d} trials | seconds min {lengths.min() / HZ:.2f}  '
          f'mean {lengths.mean() / HZ:.2f}  max {lengths.max() / HZ:.2f}')
    return obj, lengths


for block in BLOCKS:
    folder = ROOT / block
    print(f'\n=== {block}')
    if not folder.exists():
        print('  folder not found'); continue
    raw, raw_len = describe('trials-attempt.pkl (raw)', folder / 'trials-attempt.pkl')
    res, res_len = describe('trials-attempt-resampled.pkl', folder / 'trials-attempt-resampled.pkl')
    kin, kin_len = describe('all_attempt.pkl (kinematics)', folder / 'all_attempt.pkl')
    # Per movement: raw attempt lengths of the ten repetitions next to the resampled and kinematic length
    if len(raw_len) and len(raw_len) == len(res_len) and len(kin_len) * 10 == len(raw_len):
        for g in range(len(kin_len)):
            reps = raw_len[g * 10:(g + 1) * 10] / HZ
            print(f'    movement {g}: raw attempts {np.round(reps, 2).tolist()} s | resampled '
                  f'{res_len[g * 10] / HZ:.2f} s | kinematics {kin_len[g] / HZ:.2f} s')
    else:
        print(f'    (layout: raw {len(raw_len)}, resampled {len(res_len)}, kinematics {len(kin_len)} entries)')
    # Columns of the kinematic table that are not joint values (e.g. a time stamp)
    first = flatten(kin)[0] if kin is not None else None
    if hasattr(first, 'columns'):
        extra = [c for c in first.columns if not str(c).split('_')[0] in ('qpos', 'qvel', 'qacc', 'ctrl', 'sensor')]
        print('  other kinematic columns:', extra[:20])
        for c in extra:
            if 'time' in str(c).lower():
                v = np.asarray(first[c], dtype=float)
                print(f'    {c}: first {v[:3]}, step {np.median(np.diff(v)):.5f}, span {v[-1] - v[0]:.2f}')
    if (folder / 'trials-attempt-flag.pkl').exists():
        flag = load(folder / 'trials-attempt-flag.pkl')
        print('  trials-attempt-flag.pkl:', type(flag).__name__, repr(flag)[:300])
