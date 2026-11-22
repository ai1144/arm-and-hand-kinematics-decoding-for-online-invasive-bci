#!/usr/bin/env python3
"""One Ubuntu command: validate read-only inputs, train, check streams, archive results."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATES = ('20260630', '20260703', '20260707')


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.sh', '.txt', '.yaml', '.xml'}
            and not any(part in {'.venv', '__pycache__', 'results', 'tests'} for part in p.parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['smoke', 'quick', 'full'])
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/vmshare'))
    parser.add_argument('--output-root', type=Path, default=Path.home()/'thesis_results')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source = args.data_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == source or source in output.parents or source == output or output in source.parents:
        raise SystemExit('Output must be outside the read-only data root and its ancestors')
    # Exact direct-child folders: never recurse into the alternative threshold regime.
    sessions = [source/f'{date}-block{block}' for date in DATES for block in (1,2,3)]
    env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               THESIS_VM_REQUIRE_CONFIDENT_OBJECT_LABELS='1', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(ROOT))
    from decoder_core.data import _find_named_pickle, TRIAL_ATTEMPT_CANDIDATES, ALL_ATTEMPT_CANDIDATES, load_dataset
    expected_neural_features = None
    selected_files = []
    for folder in sessions:
        if not folder.is_dir() or folder.resolve().parent != source:
            raise SystemExit(f'Missing or redirected direct-child session: {folder}')
        neural = _find_named_pickle(folder, TRIAL_ATTEMPT_CANDIDATES)
        kin = _find_named_pickle(folder, ALL_ATTEMPT_CANDIDATES)
        if neural is None or kin is None:
            raise SystemExit(f'Missing attempt pickle(s) in {folder}')
        ds = load_dataset(folder)
        if len(ds.trials) != 60 or {t.repeat for t in ds.trials} != set(range(10)):
            raise SystemExit(f'Expected six groups x ten repeats in {folder}; found {len(ds.trials)} trials')
        if {t.object_id for t in ds.trials} != set(range(6)):
            raise SystemExit(f'Incomplete object mapping in {folder}')
        for t in ds.trials:
            import numpy as np
            if t.neural.shape[1] != 256 or t.qpos.shape[1] != 32 or not all(np.isfinite(a).all() for a in (t.neural,t.qpos,t.qvel)):
                raise SystemExit(f'Invalid arrays in {folder}; gesture={t.gesture}, repeat={t.repeat}; ' + '; '.join(f'{k}: shape={a.shape}, nonfinite={int((~np.isfinite(a)).sum())}' for k,a in [('neural',t.neural),('qpos',t.qpos),('qvel',t.qvel)]))
        widths = {t.neural.shape[1] for t in ds.trials}
        if len(widths) != 1:
            raise SystemExit(f"Inconsistent neural feature counts within {folder}: {widths}")
        width = widths.pop()
        if expected_neural_features is None:
            expected_neural_features = width
        if width != expected_neural_features:
            raise SystemExit(f"Neural feature count changed in {folder}: {width} != {expected_neural_features}")
        print(f"  Neural input: {width} selected features; source entries 256/257 excluded when present", flush=True)
        record = {'session': folder.name, 'trials': len(ds.trials), 'neural_features': width, 'files': []}
        for path in (neural, kin):
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    digest.update(chunk)
            record['files'].append({'path': str(path), 'bytes': path.stat().st_size,
                                    'sha256': digest.hexdigest()})
        selected_files.append(record)
        print(f'Validated {folder.name}: {neural.name} + {kin.name}', flush=True)
    if args.check_only:
        print('All nine original blocks validated. No training and no data writes.')
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    result = output/f'final_same_thresholds_{args.profile}_{stamp}'
    result.mkdir(parents=True)
    provenance = {'inputs': selected_files, 'source_sha256': source_hashes(),
                  'python': sys.version, 'platform': platform.platform(),
                  'versions': {name: importlib.metadata.version(name) for name in
                               ['numpy','pandas','scipy','scikit-learn','torch','matplotlib','PyYAML']},
                  'profile': args.profile, 'seed': 20260908, 'device': args.device,
                  'threads': args.threads, 'data_read_only': True,
                  'neural_input_selection': 'source indices 0..255 only; indices 256/257 excluded; channel order assumed, pending export documentation',
                  'changes_from_September': 'fixed onset clock; within-trial temporal metrics; validation selection; persisted classical models; PV-state PCA; corrected velocity indexing'}
    (result/'provenance.json').write_text(json.dumps(provenance, indent=2))
    command = [sys.executable, str(ROOT/'scripts/run_strict_direct_output_suite.py'),
               '--out-dir', str(result), '--profile', args.profile, '--device', args.device,
               '--seed', '20260908', '--sample-rate-hz', '100', '--export-max-trials', '12']
    for folder in sessions:
        command += ['--data-root', str(folder)]
    if args.profile == 'smoke':
        command += ['--splits','repeat_holdout','--pca-modes','none','--target-modes','qpos']
    print('Training log:', result/'training.log', flush=True)
    status = {'training': 'running', 'postprocessing': 'pending'}
    (result/'status.json').write_text(json.dumps(status,indent=2))
    try:
        with (result/'training.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['training'] = 'complete'
        with (result/'postprocess.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'summarize_results.py'),'--result-dir',str(result)],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['postprocessing'] = 'complete'
    except subprocess.CalledProcessError as e:
        status['error'] = str(e)
        (result/'status.json').write_text(json.dumps(status,indent=2))
        raise SystemExit(f'Run failed. Keep {result}; inspect training.log / postprocess.log. No success archive created.')
    (result/'status.json').write_text(json.dumps(status,indent=2))
    if os.environ.get('THESIS_SKIP_ARCHIVE') == '1':
        print('Completed:', result, '(archive skipped)', flush=True)
        return 0
    archive = result.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(result, arcname=result.name)
    print('Completed:',result, '\nDownload this archive:',archive,flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
