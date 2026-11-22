"""Load the saved results of both runs and select configurations.

Both runs are read from their own folder (``run_v1/results``, ``run_v2/results``).
Configurations are always selected on the validation posture score
(``val_qpos_R2``); the test score of the selected configuration is then reported.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RUNS = {'v1': 'run_v1', 'v2': 'run_v2'}
RUN_TITLE = {'v1': 'V1: trial-normalized time', 'v2': 'V2: fixed onset time'}
SPLITS = ['repeat_holdout', 'gesture_holdout', 'session_holdout', 'block_holdout']
SPLIT_SHORT = {'repeat_holdout': 'Repeat', 'gesture_holdout': 'Gesture',
               'session_holdout': 'Session', 'block_holdout': 'Block'}
SPLIT_LONG = {'repeat_holdout': 'New repeats of\nknown gestures', 'gesture_holdout': 'Held-out\ngestures',
              'session_holdout': 'Held-out\nsession', 'block_holdout': 'Held-out\nblock'}
OBJECTS = ['small_ball', 'big_ball', 'small_v_cylinder', 'big_v_cylinder', 'small_h_cylinder', 'big_h_cylinder']
OBJECT_LABEL = {'small_ball': 'Small ball', 'big_ball': 'Large ball',
                'small_v_cylinder': 'Small vertical\ncylinder', 'big_v_cylinder': 'Large vertical\ncylinder',
                'small_h_cylinder': 'Small horizontal\ncylinder', 'big_h_cylinder': 'Large horizontal\ncylinder'}
FAMILIES = ['ridge', 'gru', 'lfads_lite', 'transformer', 'kalman', 'refit_kalman']
DIRECT = ['ridge', 'gru', 'lfads_lite', 'transformer']
FAMILY_LABEL = {'ridge': 'Ridge', 'gru': 'GRU', 'lfads_lite': 'LFADS-lite', 'transformer': 'Transformer',
                'kalman': 'Kalman', 'refit_kalman': 'ReFIT Kalman'}
TARGET_LABEL = {'qpos': 'Position', 'qvel': 'Velocity', 'pv': 'Position + velocity'}


def load_run(root, run):
    """Return the saved tables of one run as a dictionary of DataFrames."""
    folder = Path(root) / RUNS[run] / 'results'
    board = pd.read_csv(folder / 'leaderboard.csv')
    board['condition_object'] = board.condition_object.astype(bool)
    board = board[board.split_mode.isin(SPLITS)].copy()        # the four reported holdouts
    trials = pd.read_csv(folder / 'trial_metrics.csv.gz')
    trials['condition_object'] = trials.condition_object.astype(bool)
    return dict(run=run, folder=folder, board=board, trials=trials,
                joints=pd.read_csv(folder / 'per_dimension_metrics.csv'),
                history=pd.read_csv(folder / 'training_history.csv'),
                index=pd.read_csv(folder / 'combined_trial_index.csv'))


def select(board, groups, score='val_qpos_R2'):
    """Best configuration per group, chosen on the validation score."""
    ranked = board.sort_values([score, 'experiment_id'], ascending=[False, True])
    return ranked.groupby(groups, dropna=False).head(1)


def candidate(r, object_id=True, split='repeat_holdout', family=None):
    """Validation-selected configuration for one split and cue condition."""
    b = r['board']
    part = b[b.split_mode.eq(split) & b.condition_object.eq(object_id)]
    if family is not None:
        part = part[part.model_family.eq(family)]
    return select(part, ['split_mode']).iloc[0]


def pca_tag(row):
    """Short label of the dimensionality-reduction setting, e.g. 'N64 / K12'."""
    neural = 'N64' if str(row.pca_mode).startswith('pca') else 'N0'
    target = str(row.target_mode)
    kin = 'S12' if 'statepca' in target else ('K12' if 'pca' in target else 'K0')
    return f'{neural} / {kin}'


def describe(row):
    """Readable one-line description of a configuration."""
    target = TARGET_LABEL[row.target_base_mode].lower()
    neural = 'neural PCA64' if str(row.pca_mode).startswith('pca') else 'no neural PCA'
    kin = 'kinematic PCA12' if 'pca' in str(row.target_mode) else 'no kinematic PCA'
    cue = 'object ID' if row.condition_object else 'no object ID'
    return f'{FAMILY_LABEL[row.model_family]}, {cue}, {neural}, {target} target, {kin}'


def test_trials(r, row, variant='raw'):
    """Per-trial test metrics of one configuration ('raw' or '1euro' output)."""
    t = r['trials']
    return t[t.experiment_id.eq(row.experiment_id) & t.output_variant.eq(variant)]
