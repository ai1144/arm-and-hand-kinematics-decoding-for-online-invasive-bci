# V1: trial-normalized time

The V1 pipeline gives the decoders elapsed time as `t / (T - 1)`, the relative position within a cued trial.

This source-only release contains [functions/](functions/): training, evaluation, data loading, decoder definitions and streaming/replay integration. See the [training instructions](functions/README.md).

The analysis notebook expects externally generated V1 artifacts in `models/` and `results/`, including leaderboard tables, per-trial metrics, training histories, provenance and replay arrays as required by that run. Those artifacts, recording blocks, supplied reference templates and third-party simulator assets are not distributed in this repository. The saved notebook outputs and selected README figures illustrate the original experiment.
