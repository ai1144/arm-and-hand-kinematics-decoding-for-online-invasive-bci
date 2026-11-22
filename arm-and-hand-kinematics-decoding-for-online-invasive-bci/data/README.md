# Recording inputs

This folder contains the project's [trial-duration summary script](print_trial_durations.py). Recording files and generated summaries are excluded from this source-only release.

For raw-trial inspection or retraining, supply recording blocks locally under `data/patient_sessions/` and configure the data-root argument or notebook `RECORDINGS` parameter. The thesis experiment used nine blocks across three recording days, with three blocks per day. Each block requires the files expected by the selected loader, including `trials-attempt-resampled.pkl` and `all_attempt.pkl`.

`print_trial_durations.py` is intended to run on the recording machine; configure its `ROOT` path for that environment. It generates `recorded_attempt_durations.csv`, which the notebook expects in this folder for its duration analysis. Neither the recordings nor that CSV are included in this repository.
