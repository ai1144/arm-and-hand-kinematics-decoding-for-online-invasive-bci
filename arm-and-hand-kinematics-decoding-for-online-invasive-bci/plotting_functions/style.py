"""Shared colours, fonts and figure saving for all thesis plots."""
from pathlib import Path
import matplotlib

FAMILY_COLOR = {'ridge': '#4e79a7', 'gru': '#2ca58d', 'lfads_lite': '#f28e2b', 'transformer': '#7b6bb7',
                'kalman': '#8f8f8f', 'refit_kalman': '#c44e52'}
RUN_COLOR = {'v1': '#005293', 'v2': '#e37222'}            # TUM blue / TUM orange
OBJECT_COLOR = {'small_ball': '#4e79a7', 'big_ball': '#a0cbe8', 'small_v_cylinder': '#59a14f',
                'big_v_cylinder': '#8cd17d', 'small_h_cylinder': '#b07aa1', 'big_h_cylinder': '#d4a6c8'}
LIMB_COLOR = {'shoulder': '#4e79a7', 'elbow': '#76b7b2', 'wrist': '#59a14f', 'thumb': '#f28e2b',
              'index': '#e15759', 'middle': '#b07aa1', 'ring finger': '#edc948', 'little finger': '#9c755f'}
RAW_GREY = '#9aa0a6'
FILTER_GREEN = '#2ca58d'


def setup():
    """Consistent, print-friendly defaults (vector fonts in PDFs)."""
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def limb(name):
    """Limb group of a posture coordinate name."""
    for prefix, group in [('Shoulder', 'shoulder'), ('Elbow', 'elbow'), ('Wrist', 'wrist'), ('T_', 'thumb'),
                          ('I_', 'index'), ('M_', 'middle'), ('Ring', 'ring finger'), ('Little', 'little finger')]:
        if name.startswith(prefix):
            return group
    raise ValueError(name)


def save(fig, folder, name, source=None, dpi=300):
    """Save a figure as vector PDF and high-resolution PNG, plus its source data.

    The figure is left open, so that the notebook displays it below the cell that drew it.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder / f'{name}.png', dpi=dpi, bbox_inches='tight')
    if source is not None:
        source.to_csv(folder / f'{name}_source.csv', index=False)
