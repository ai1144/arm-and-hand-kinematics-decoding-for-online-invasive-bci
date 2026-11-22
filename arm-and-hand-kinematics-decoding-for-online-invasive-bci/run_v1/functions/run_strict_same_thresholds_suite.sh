#!/usr/bin/env bash
set -euo pipefail

# strict same-threshold vm launcher
#
# usage:
#   bash run_strict_same_thresholds_suite.sh smoke cpu
#   bash run_strict_same_thresholds_suite.sh quick auto
#   bash run_strict_same_thresholds_suite.sh full auto
#
# this run uses only same-threshold session folders directly under /mnt/vmshare.
# it ignores the nested "thresholds defined per block" folder and excludes the
# older mac-uploaded patient_sessions by using --new-only.

PROFILE="${1:-full}"
DEVICE="${2:-auto}"
DATA_ROOT="${THESIS_VM_DATA_ROOT:-/mnt/vmshare}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$(pwd)/vm_results/patient_100hz_same_thresholds_strict_${PROFILE}_${TIMESTAMP}"

SESSION_DATE_ARGS=()
if [[ -n "${THESIS_SESSION_HOLDOUT_TEST_DATE:-}" ]]; then
  SESSION_DATE_ARGS=(--session-holdout-test-date "${THESIS_SESSION_HOLDOUT_TEST_DATE}")
fi

BLOCK_HOLDOUT_TEST_BLOCK="${THESIS_BLOCK_HOLDOUT_TEST_BLOCK:-3}"
EXPORT_REPEAT="${THESIS_EXPORT_REPEAT:-8}"
EXPORT_MAX_TRIALS="${THESIS_EXPORT_MAX_TRIALS:-12}"

SPLIT_ARGS=(repeat_holdout gesture_holdout session_holdout block_holdout)
PCA_ARGS=(none pca)
TARGET_ARGS=(qpos qvel pv qpos_pca qvel_pca pv_pca)
MAX_EXPERIMENT_ARGS=()
if [[ "${PROFILE}" == "smoke" ]]; then
  SPLIT_ARGS=(repeat_holdout)
  PCA_ARGS=(none)
  TARGET_ARGS=(qpos)
  MAX_EXPERIMENT_ARGS=(--max-experiments 2)
fi

export THESIS_VM_REQUIRE_CONFIDENT_OBJECT_LABELS=1

echo "Threshold condition: same_thresholds"
echo "Data root: ${DATA_ROOT}"
echo "Ignored folder: ${DATA_ROOT}/thresholds defined per block"
echo "Training mode: strict direct, no hybrid/template kinematic input"
echo "Object mapping: handle_id -> geom_id -> canonical six-object name"
echo "Splits: ${SPLIT_ARGS[*]}"
echo "Neural PCA modes: ${PCA_ARGS[*]}"
echo "Target modes: ${TARGET_ARGS[*]}"
echo "MuJoCo export: repeat ${EXPORT_REPEAT}, max ${EXPORT_MAX_TRIALS} trials per model"
echo "Output: ${OUT_ROOT}"

python scripts/run_vm_split_pca_suite.py \
  --suite-script scripts/run_strict_direct_output_suite.py \
  --data-search-root "${DATA_ROOT}" \
  --include-date 20260630 \
  --include-date 20260703 \
  --include-date 20260707 \
  --new-only \
  --profile "${PROFILE}" \
  --device "${DEVICE}" \
  --sample-rate-hz 100 \
  --validate-data \
  --make-archive \
  --out-root "${OUT_ROOT}" \
  --max-depth 3 \
  --splits "${SPLIT_ARGS[@]}" \
  --pca-modes "${PCA_ARGS[@]}" \
  --pca-components 64 \
  --target-modes "${TARGET_ARGS[@]}" \
  --qpos-pca-components 12 \
  --block-holdout-test-block "${BLOCK_HOLDOUT_TEST_BLOCK}" \
  --export-repeat "${EXPORT_REPEAT}" \
  --export-max-trials "${EXPORT_MAX_TRIALS}" \
  "${MAX_EXPERIMENT_ARGS[@]}" \
  "${SESSION_DATE_ARGS[@]}"
