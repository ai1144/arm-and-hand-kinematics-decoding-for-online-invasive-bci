# V1 training code (trial-normalized time)

**Source-only release:** recordings, trained models, saved result datasets and the external `new_info/` assets/templates are not included. Supply compatible external inputs before training or rendering simulator replays. The commands below describe the full experiment environment.

This folder contains the code that trained the V1 benchmark, whose saved results are expected in the external `../results` directory. In V1, the elapsed-time input of the decoders is the position within the cued trial, *t / (T − 1)*. The thesis notebook loads the saved results; this code is only needed to retrain.

## Start the training

From the repository root, with the environment of the notebook active and the nine recording blocks in `data/patient_sessions`:

```bash
cd run_v1/functions
THESIS_VM_DATA_ROOT=../../data/patient_sessions THESIS_SESSION_HOLDOUT_TEST_DATE=20260707 \
    bash run_strict_same_thresholds_suite.sh full auto
```

- `full` trains the complete grid of 432 configurations: 4 holdouts; Ridge, GRU, LFADS-lite and Transformer with object cue on/off, neural PCA on/off and 6 targets (posture, velocity or both, each with and without kinematic PCA); Kalman and ReFIT-style Kalman without object cue or time input.
- `smoke` instead of `full` trains two models on the repeat holdout, which checks the whole pipeline within minutes.
- `auto` uses a GPU if one is available and the CPU otherwise (`cpu` forces the CPU).
- `THESIS_SESSION_HOLDOUT_TEST_DATE=20260707` holds out the third day in the session holdout. This is also the default; setting it explicitly only avoids a problem of the old bash version of macOS.

The results are written to a new, time-stamped folder `vm_results/patient_100hz_same_thresholds_strict_<profile>_<time>/`. The saved results in `../results` are never overwritten. The same two commands can be started from the notebook (Section 4.3).

The streaming tests of Appendix A (artificial input and replay of recorded trials through a single-bin decoder node) are run afterwards with:

```bash
bash run_brand_node_tests_from_latest_same_thresholds.sh vm_results/<result folder>
```

## Files

| File | Role |
|---|---|
| `run_strict_same_thresholds_suite.sh` | Entry point: sets the grid (holdouts, PCA, targets) and starts the training |
| `scripts/run_vm_split_pca_suite.py` | Finds the nine blocks, checks the data, runs the suite and packs the results |
| `scripts/discover_sessions.py` | Finds and summarizes the block folders |
| `scripts/run_strict_direct_output_suite.py` | Trains and evaluates the decoders from neural input, elapsed time and object cue only |
| `scripts/run_split_pca_model_suite.py` | Splits, PCA, decoder families, metrics and result tables |
| `scripts/run_multisession_online_decoding.py`, `run_extra_experiments.py`, `train_advanced_policies.py`, `train_smooth_decoders.py` | Shared helpers (data handling, Kalman and recurrent models, output smoothing) |
| `scripts/audit_strict_no_kinematic_inputs.py` | Check, run at the end, that no reference kinematics were given to a decoder as input |
| `scripts/view_mujoco_hand.py` | MuJoCo replay of reference and decoded movement |
| `decoder_core/` | Data loading, models, metrics and the object mapping of the MuJoCo scene |
| `new_info/` (external; excluded) | Compatible MuJoCo arm/hand model, object assets and supplied movement references; obtain and configure these separately. |
| `brand_streaming_decoder_node.py`, `run_brand_node_tests_from_latest_same_thresholds.sh`, `scripts/plot_brand_node_results.py` | Streaming tests of Appendix A |
