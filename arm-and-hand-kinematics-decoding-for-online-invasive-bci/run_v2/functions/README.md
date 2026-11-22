# V2 training code (fixed onset clock)

**Source-only release:** recordings, trained models, saved result datasets and the external `new_info/` assets/templates are not included. Supply compatible external inputs before training or rendering simulator replays. The commands below describe the full experiment environment.

This folder contains the code that trained the V2 benchmark, whose saved results are expected in the external `../results` directory. In V2, the elapsed-time input of the decoders is a clock started at trial onset, *t / 600* (6 s at 100 Hz), which does not depend on the duration of the trial. Data, splits, decoder families and settings are otherwise those of V1. The thesis notebook loads the saved results; this code is only needed to retrain.

## Start the training

From the repository root, with the environment of the notebook active and the nine recording blocks in `data/patient_sessions`:

```bash
python run_v2/functions/run_final.py full --data-root data/patient_sessions --output-root new_training_results --device cpu
```

- `full` trains the complete grid of 480 configurations (4 holdouts; Ridge, GRU, LFADS-lite and Transformer with object cue on/off, neural PCA on/off and 6 targets; Kalman and ReFIT-style Kalman without object cue or time input, with and without state PCA).
- `smoke` instead of `full` trains every decoder family once on the repeat holdout (posture target, no PCA) with two training epochs, which checks the whole pipeline within minutes; `run_final.py smoke --check-only` only checks the data.
- `--device cuda` uses a GPU. Results on other devices can differ slightly in the last digits.

Every call writes a new, time-stamped result folder (and a `.tar.gz` archive of it) into `--output-root`. The saved results in `../results` are never overwritten, and the recordings are only read. The same command can be started from the notebook (Section 4.3).

## Files

| File | Role |
|---|---|
| `run_final.py` | Entry point: checks the nine blocks, trains the grid, runs the streaming checks and packs the results |
| `scripts/run_strict_direct_output_suite.py` | Trains and evaluates the decoders from neural input, elapsed time and object cue only |
| `scripts/run_split_pca_model_suite.py` | Causal preprocessing, splits, PCA, decoder families and metrics |
| `scripts/run_multisession_online_decoding.py` | Loads the attempted-movement trials of the nine blocks with their object labels |
| `scripts/train_smooth_decoders.py` | Causal output smoothing, reset at the start of each trial |
| `scripts/view_mujoco_hand.py` | MuJoCo replay of reference and decoded movement |
| `summarize_results.py` | Validation-selected tables, streaming checks, neural controls and plots (called by `run_final.py`) |
| `streaming.py`, `brand_streaming_decoder_node.py` | Single-bin decoder for every saved model family, used for the streaming checks |
| `decoder_core/` | Data loading, models, metrics and the object mapping of the MuJoCo scene |
| `new_info/` (external; excluded) | Compatible MuJoCo arm/hand model, object assets and supplied movement references; obtain and configure these separately. |
| `tests/test_protocol.py` | Numerical checks of the Kalman update and of the smoothness metric (no patient data needed) |
