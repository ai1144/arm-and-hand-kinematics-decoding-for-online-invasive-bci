#!/usr/bin/env bash
set -euo pipefail

# Test deployable no-PCA/direct decoder artifacts from the latest same-threshold
# full run.  This is a pre-BRAND artificial stream test: it does not require the
# official BRAND repository and does not measure decoding accuracy from real
# online data.  It checks that the node loads, respects the decode flag, emits
# 32-D qpos only during active decode, maps optional object IDs, and reports
# per-bin latency.

RESULT_DIR="${1:-}"
DEVICE="${2:-cpu}"
SESSION_ROOT="${3:-}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="$(ls -td vm_results/patient_100hz_same_thresholds_strict_full_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$RESULT_DIR" || ! -d "$RESULT_DIR" ]]; then
  echo "No strict same-threshold full result folder found. Run: bash run_strict_same_thresholds_suite.sh full auto" >&2
  exit 1
fi

echo "Using result folder: $RESULT_DIR"
echo "Using device: $DEVICE"

ARTIFACTS=(
  "repeat_holdout__none__gru__direct__obj__target_qpos/gru_direct_obj_qpos.pt"
  "repeat_holdout__none__gru__direct__nolabel__target_qpos/gru_direct_nolabel_qpos.pt"
  "repeat_holdout__none__lfads_lite__direct__obj__target_qpos/lfads_lite_direct_obj_qpos.pt"
  "repeat_holdout__none__lfads_lite__direct__nolabel__target_qpos/lfads_lite_direct_nolabel_qpos.pt"
  "repeat_holdout__none__transformer__direct__obj__target_qpos/transformer_direct_obj_qpos.pt"
  "repeat_holdout__none__transformer__direct__nolabel__target_qpos/transformer_direct_nolabel_qpos.pt"
)

for REL in "${ARTIFACTS[@]}"; do
  ARTIFACT="$RESULT_DIR/models/$REL"
  if [[ ! -f "$ARTIFACT" ]]; then
    echo "Skipping missing artifact: $ARTIFACT"
    continue
  fi
  SAFE_NAME="$(echo "$REL" | tr '/.' '__')"
  OUT_CSV="$RESULT_DIR/brand_node_artificial_${SAFE_NAME}.csv"
  echo
  echo "Artificial stream node test: $ARTIFACT"
  "$PYTHON_BIN" brand_streaming_decoder_node.py \
    --artifact "$ARTIFACT" \
    --object-id 56 \
    --device "$DEVICE" \
    --sample-rate-hz 100 \
    --decode-start 100 \
    --decode-stop 500 \
    --out-csv "$OUT_CSV"
done

if [[ -n "$SESSION_ROOT" ]]; then
  ARTIFACT="$RESULT_DIR/models/repeat_holdout__none__gru__direct__obj__target_qpos/gru_direct_obj_qpos.pt"
  if [[ ! -f "$ARTIFACT" ]]; then
    ARTIFACT="$RESULT_DIR/models/repeat_holdout__none__gru__direct__nolabel__target_qpos/gru_direct_nolabel_qpos.pt"
  fi
  if [[ ! -f "$ARTIFACT" ]]; then
    echo "No GRU artifact found for offline session replay." >&2
    exit 1
  fi
  echo
  echo "Offline VM session replay through node core: $SESSION_ROOT"
  "$PYTHON_BIN" brand_streaming_decoder_node.py \
    --artifact "$ARTIFACT" \
    --offline-session-root "$SESSION_ROOT" \
    --offline-repeat 8 \
    --device "$DEVICE" \
    --sample-rate-hz 100 \
    --out-dir "$RESULT_DIR/brand_node_vm_replay"
fi

echo
echo "Done. Node-test CSV/JSON outputs were written inside:"
echo "  $RESULT_DIR"

"$PYTHON_BIN" scripts/plot_brand_node_results.py --result-dir "$RESULT_DIR"
