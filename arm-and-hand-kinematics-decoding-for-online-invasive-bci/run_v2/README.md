# V2: fixed onset time

The V2 pipeline gives the decoders elapsed time as `t / 600`, a clock started at trial onset with a six-second scale at 100 Hz.

This source-only release contains [functions/](functions/): training, evaluation, data loading, decoder definitions and streaming/replay integration. The main training entry point is `functions/run_final.py`. See the [training instructions](functions/README.md).

The analysis notebook expects externally generated V2 artifacts in `models/` and `results/`, including leaderboard tables, per-trial metrics, training histories, provenance and replay arrays as required by that run. Those artifacts, recording blocks, supplied reference templates and third-party simulator assets are not distributed in this repository. The saved notebook outputs and selected README figures illustrate the original experiment.
