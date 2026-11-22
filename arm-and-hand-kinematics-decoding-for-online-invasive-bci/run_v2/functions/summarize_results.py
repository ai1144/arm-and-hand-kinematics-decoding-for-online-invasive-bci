#!/usr/bin/env python3
"""Validation-selected tables, stream parity/timing checks, controls, and plots."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
from run_multisession_online_decoding import combine_sessions
from run_split_pca_model_suite import make_split, temporal_metrics_by_trial, NeuralProjector
from run_strict_direct_output_suite import train_sequence_direct, fit_target_codec, qpos_predictions_from_targets
from streaming import Decoder


def stream_trial(artifact, trial, neural=None):
    decoder = Decoder(artifact, initial_qpos=trial.qpos[0])
    values = trial.neural if neural is None else neural
    assert decoder.step(values[0], decode_enabled=False, timestamp_s=-.01, object_id=trial.object_id) is None
    preds=[]; timing=[]
    for i,x in enumerate(values):
        step=decoder.step(x, decode_enabled=True, timestamp_s=i/100., object_id=trial.object_id)
        if step is None or step.qpos.shape != (32,) or not np.isfinite(step.qpos).all():
            raise RuntimeError('Stream emitted an invalid posture packet')
        preds.append(step.qpos);timing.append(step.latency_ms)
    assert decoder.step(values[-1], decode_enabled=False, timestamp_s=len(values)/100., object_id=trial.object_id) is None
    restarted=decoder.step(values[0], decode_enabled=True, timestamp_s=0., object_id=trial.object_id)
    if not np.allclose(restarted.qpos,preds[0],atol=1e-5,rtol=1e-5):
        raise RuntimeError('Decoder did not reset reproducibly at trial onset')
    return np.asarray(preds),np.asarray(timing)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    args=parser.parse_args();out=args.result_dir.resolve()
    manifest=json.loads((out/'run_manifest.json').read_text())
    leaderboard=pd.read_csv(out/'leaderboard.csv')
    if leaderboard.experiment_id.duplicated().any():
        raise RuntimeError('Duplicated experiment IDs')
    specs = sum(1 if f in {'kalman','refit_kalman'} else len(manifest['condition_modes'])
                for f in manifest['model_families'])
    expected = len(manifest['splits']) * len(manifest['pca_modes']) * len(manifest['target_modes']) * specs
    if len(leaderboard)!=expected:
        raise RuntimeError(f'Incomplete grid: expected {expected}, found {len(leaderboard)}')
    # Test scores are never used in selection or sorting these tables.
    rank=leaderboard.sort_values('val_qpos_R2',ascending=False)
    selected=rank.groupby(['split_mode','model_family','condition_object'],sort=False).head(1)
    selected.to_csv(out/'validation_selected_by_family.csv',index=False)

    best=rank.groupby('split_mode',sort=False).head(1)
    best.to_csv(out/'validation_selected_best_by_split.csv',index=False)
    dataset,_=combine_sessions([Path(p) for p in manifest['data_roots']],ROOT/'new_info/data',out)
    trial_metrics=pd.read_csv(out/'trial_metrics.csv')
    raw=trial_metrics[trial_metrics.output_variant.eq('raw')]
    candidate=best[best.split_mode.eq('repeat_holdout')].iloc[0]
    candidate_trials=raw[raw.experiment_id.eq(candidate.experiment_id)]
    summary=candidate_trials.groupby('session_date').agg(
        n_trials=('qpos_R2','count'),R2_mean=('qpos_R2','mean'),R2_std=('qpos_R2','std'),
        CC_mean=('qpos_CC','mean'),CC_std=('qpos_CC','std'),
        RMSE_mean=('qpos_RMSE','mean'),RMSE_std=('qpos_RMSE','std'))
    summary.to_csv(out/'selected_model_session_variability.csv')
    timing_rows=[];stream_dir=out/'brand_stream_checks';stream_dir.mkdir(exist_ok=True)
    splits={}
    for split in manifest['splits']:
        splits[split]=make_split(dataset.trials,split,20260908)
    for _,row in selected.iterrows():
        test=splits[row.split_mode][2]
        trial=test[0]
        archive=np.load(row.prediction_archive)
        n=int(archive['lengths'][0]);expected_pred=archive['qpos_pred'][:n]
        pred,ms=stream_trial(row.artifact,trial)
        delta=float(np.max(np.abs(pred-expected_pred)))
        # Numerical differences from single-bin GEMM versus whole-sequence GEMM.
        passed=bool(np.allclose(pred,expected_pred,rtol=2e-4,atol=2e-4))
        record=dict(experiment_id=row.experiment_id,model_family=row.model_family,
                    n_active_bins=len(ms),max_abs_stream_offline_difference=delta,parity_passed=passed,
                    median_step_ms=float(np.median(ms)),p95_step_ms=float(np.percentile(ms,95)),
                    p99_step_ms=float(np.percentile(ms,99)),max_step_ms=float(ms.max()),
                    bins_over_10ms=int((ms>10).sum()),first_step_ms=float(ms[0]),
                    timing_scope='preprocessing + model + reconstruction; excludes Redis and actuator I/O')
        timing_rows.append(record)
        np.savez_compressed(stream_dir/(row.experiment_id+'.npz'),qpos_pred=pred,qpos_true=trial.qpos,latency_ms=ms)
        print('Stream check:',row.experiment_id,'parity',passed,'p95 ms',record['p95_step_ms'],flush=True)
    pd.DataFrame(timing_rows).to_csv(out/'brand_stream_latency_and_parity.csv',index=False)
    # Circular shift preserves each trial's channels/covariance and marginal samples.
    train,val,test,_=splits['repeat_holdout']
    control_rows=[]
    for mode in ['real','circular_time_shift','training_mean_neural']:
        preds=[]
        mean=np.concatenate([t.neural for t in train]).mean(axis=0)
        for t in test:
            neural=t.neural if mode=='real' else (np.roll(t.neural,max(1,len(t.neural)//2),axis=0)
                    if mode=='circular_time_shift' else np.broadcast_to(mean,t.neural.shape).copy())
            pred,_=stream_trial(candidate.artifact,t,neural);preds.append(pred)
        metrics=temporal_metrics_by_trial([t.qpos for t in test],preds,100.)
        control_rows.append(dict(control=mode,experiment_id=candidate.experiment_id,**metrics))
    pd.DataFrame(control_rows).to_csv(out/'selected_model_matched_neural_controls.csv',index=False)
    # A separately trained nonlinear timing/object baseline tests task predictability.
    mean=np.concatenate([t.neural for t in train]).mean(axis=0)
    def mean_trials(trials):
        return [replace(t,neural=np.broadcast_to(mean,t.neural.shape).copy()) for t in trials]
    baseline_dir=out/'task_only_baseline';baseline_dir.mkdir(exist_ok=True)
    codec=fit_target_codec(train,requested_mode='qpos',components=12,seed=20260908)
    row,preds,_,history=train_sequence_direct(mean_trials(train),mean_trials(val),mean_trials(test),
          NeuralProjector(mode='none'),6,model_family='gru',condition_object=True,target_codec=codec,
          device_name='cpu',profile=manifest['profile'],seed=20260908,bidirectional_lfads=False,
          gru_layers=1,sample_rate_hz=100.,out_dir=baseline_dir)
    row['description']='Separately trained phase+object GRU; neural channels fixed to training mean'
    (baseline_dir/'metrics.json').write_text(json.dumps(row,indent=2))
    history.to_csv(baseline_dir/'training_history.csv',index=False)
    pd.DataFrame([row]).to_csv(out/'task_only_baseline.csv',index=False)
    plot_dir=out/'plots';plot_dir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(summary.index.astype(str),summary.R2_mean,yerr=summary.R2_std,capsize=5)
    ax.set(ylabel='Mean trial qpos R² ± sample SD',xlabel='Recording date',title='Validation-selected repeat-holdout model')
    fig.tight_layout();fig.savefig(plot_dir/'selected_session_variability.png',dpi=180);plt.close(fig)
    for metric in ['test_qpos_R2','test_qpos_CC','test_qpos_RMSE']:
        fig,ax=plt.subplots(figsize=(11,5))
        for (family,cue),group in selected.groupby(['model_family','condition_object']):
            ax.plot(group.split_mode,group[metric],marker='o',label=f'{family}, object={bool(cue)}')
        ax.set_ylabel(metric);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(plot_dir/(metric+'_validation_selected.png'),dpi=180);plt.close(fig)
    issues=[r['experiment_id'] for r in timing_rows if not r['parity_passed']]
    notes=f'''# Final-run interpretation

Completed grid: {len(leaderboard)} configurations. All ranking tables select by
validation qpos R2. The default phase feature is elapsed bins / 600, independent
of actual final trial duration. The raw input files are recorded with hashes in
provenance.json. Results differ methodologically from the September benchmark.

R2/CC/RMSE are pooled per-dimension accuracy metrics. Timing/jitter/lag are
computed within trials and then averaged equally across trials. Onset fields are
heuristics, not physiological event measurements; inspect initial transients.
The legacy procrustes_corr is centered normalized correlation without rotation.
orthogonal_procrustes_corr additionally permits orthogonal alignment.

Kalman *_pca configurations reduce the TRAINING PV state to 12 PCs and reconstruct
PV before selecting qpos/qvel; this is state PCA, not a neural-network output-head
PCA. Kalman receives neural observations only, with no object/phase input.
For qvel integration, the known initial simulator posture is explicitly used.

BRAND stream checks use actual held-out neural trials, gating and state resets.
Latency excludes Redis I/O and hardware/actuator delays. Timing failures do not
change accuracy metrics. Single-trial parity failures: {issues}.

predictions/ saves all test target and posture arrays per model for recomputation.
mujoco_compare/ and mujoco_compare_1euro/ contain a consistent subset of up to
12 replay trials per configuration; they are visual checks, not the full test set.
task_only_baseline.csv is a separately trained nonlinear task-clock/object ablation.
The matched controls complement, but do not prove absence of, neural leakage.
'''
    (out/'RESULTS_README.md').write_text(notes)
    if issues:
        raise RuntimeError(f'Stream/offline parity failed for {issues}; preserve results and investigate')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
