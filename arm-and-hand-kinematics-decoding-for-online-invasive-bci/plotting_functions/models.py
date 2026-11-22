"""Load trained decoders from disk (no training).

Each run keeps its checkpoints in ``run_vX/models/<configuration>/``. Neural-network
checkpoints (.pt) store the weights together with the preprocessing fitted on the
training split (input/target standardization, neural PCA and kinematic PCA).
"""
import importlib.util
import sys
from pathlib import Path
import torch
from .results import RUNS


def forget_run_code(root):
    """Both runs ship modules with the same names (decoder_core and the training scripts). Remove any that are
    already loaded, so that the next import uses the code of the intended run."""
    folders = [str((Path(root) / RUNS[r] / 'functions').resolve()) for r in RUNS]
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        inside = bool(file) and any(str(Path(file).resolve()).startswith(f) for f in folders)
        if name == 'decoder_core' or name.startswith('decoder_core.') or inside:
            del sys.modules[name]


def _architectures(root, run):
    """The model definitions of the run's own training code."""
    forget_run_code(root)
    path = Path(root) / RUNS[run] / 'functions/scripts/run_split_pca_model_suite.py'
    sys.path.insert(0, str(path.parents[1]))
    spec = importlib.util.spec_from_file_location(f'{run}_suite', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SequenceNetFactory


def _numpy_compatibility():
    """Allow checkpoints written with NumPy 2 to be read with NumPy 1.x."""
    import numpy
    if not hasattr(numpy, '_core'):
        import numpy.core.multiarray, numpy.core.numeric
        sys.modules.setdefault('numpy._core', numpy.core)
        sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', numpy.core.numeric)


def load_decoder(root, run, experiment_id):
    """Rebuild one trained GRU / LFADS-lite / Transformer and load its weights."""
    _numpy_compatibility()
    files = list((Path(root) / RUNS[run] / 'models' / experiment_id).glob('*.pt'))
    if len(files) != 1:
        raise FileNotFoundError(f'Expected one checkpoint in {experiment_id}')
    payload = torch.load(files[0], map_location='cpu', weights_only=False)
    factory = _architectures(root, run)
    model = factory.make(torch, torch.nn, model_family=payload['model_family'], input_dim=len(payload['x_mean']),
                         output_dim=payload['output_dim'], hidden_dim=payload['hidden_dim'],
                         factor_dim=payload.get('factor_dim', 32), bidirectional_lfads=False,
                         gru_layers=payload.get('gru_layers', 1))
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload


def checkpoint_inventory(root, run):
    """Number of saved model folders per split and model family."""
    import pandas as pd
    rows = []
    for folder in sorted((Path(root) / RUNS[run] / 'models').iterdir()):
        if folder.is_dir():
            split, pca, family = folder.name.split('__')[:3]
            rows.append(dict(split=split, neural_pca=pca, family=family,
                             files=', '.join(sorted(p.suffix for p in folder.iterdir()))))
    return pd.DataFrame(rows)
