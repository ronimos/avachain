#!/bin/bash
# expand_march7_envelopes.sh — Generate probability envelope scenarios for
# 2026-03-07 conditions at increasing size-factor widths.
#
# All envelopes use the same March 7 snowpack state.  T_0 through T_6
# represent progressively wider uncertainty bounds — not different forecast
# days.  Each horizon adds outer size factors to the previous envelope:
#
#   T_0:  0.85 1.00                              →  30 scenarios
#   T_1:  0.70 … 1.30  (0.15-step)              →  75 scenarios
#   T_2:  0.55 … 1.45  (0.15-step)              → 105 scenarios
#   T_3:  0.40 … 1.60  (0.15-step)              → 135 scenarios
#   T_4:  0.25 … 2.00  (0.25-step)              → 120 scenarios
#   T_5:  0.25 … 2.50  (0.25-step)              → 150 scenarios
#   T_6:  0.25 … 3.00  (0.25-step)              → 180 scenarios
#
# T_0–T_3 use 0.15 steps (fine resolution near the observed state).
# T_4–T_6 switch to 0.25 steps (coarser, extended uncertainty envelope).
#
# Outputs all go to: outputs/scenarios/2026-03-07/T_{0..6}/
#
# Prerequisite: 2026-03-07 analyze step must already have run
# (all_start_zone_features_2026-03-07.csv must exist).

set -euo pipefail

PROJECT_DIR=/home/ron/snowpack_model_feeder
ANALYSIS="$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/src/avachain/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/expand_march7_envelopes_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  March 7 Probability Envelopes (T_0 – T_6)"
echo "  Started: $(date)"
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

FEAT_CSV_ALL="$PROJECT_DIR/outputs/analysis/all_start_zone_features_2026-03-07.csv"
FEAT_CSV_GRP="$PROJECT_DIR/outputs/analysis/release_zone_features_2026-03-07.csv"
if [[ ! -f "$FEAT_CSV_ALL" && ! -f "$FEAT_CSV_GRP" ]]; then
    echo "ERROR: no features CSV found for 2026-03-07." >&2
    echo "       Run 'analyze --snapshot-date 2026-03-07' first." >&2
    exit 1
fi
if [[ ! -f "$FEAT_CSV_ALL" ]]; then
    echo "WARNING: using group-level features (release_zone_features_2026-03-07.csv)."
    echo "         Rerun analyze to get full start zone coverage."
    echo ""
fi

SECONDS=0

echo ">>> Generating T_0 – T_6 envelopes from 2026-03-07 snowpack state (parallel)"
echo "    T_0:  0.85 1.00                           (30 scenarios)"
echo "    T_1:  0.70 … 1.30  0.15-step              (75 scenarios)"
echo "    T_2:  0.55 … 1.45  0.15-step             (105 scenarios)"
echo "    T_3:  0.40 … 1.60  0.15-step             (135 scenarios)"
echo "    T_4:  0.25 … 2.00  0.25-step             (120 scenarios)"
echo "    T_5:  0.25 … 2.50  0.25-step             (150 scenarios)"
echo "    T_6:  0.25 … 3.00  0.25-step             (180 scenarios)"
echo ""

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_0 \
    --size-factors 0.85 1.00 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_1 \
    --size-factors 0.70 0.85 1.00 1.15 1.30 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_2 \
    --size-factors 0.55 0.70 0.85 1.00 1.15 1.30 1.45 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_3 \
    --size-factors 0.40 0.55 0.70 0.85 1.00 1.15 1.30 1.45 1.60 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_4 \
    --size-factors 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_5 \
    --size-factors 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_6 \
    --size-factors 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

wait
gen_elapsed=$SECONDS
echo ""
echo "    Generation done: $(($gen_elapsed / 60))m $(($gen_elapsed % 60))s"
echo ""

# --- Summary ---
total_elapsed=$SECONDS
OUT_BASE="$PROJECT_DIR/outputs/scenarios/2026-03-07"
echo "============================================================"
echo "  Done"
echo "  Finished: $(date)"
echo "  Total runtime: $(($total_elapsed / 60))m $(($total_elapsed % 60))s"
echo ""
echo "  Outputs (all under $OUT_BASE):"
echo "    T_0:  30  scenarios"
echo "    T_1:  75  scenarios"
echo "    T_2:  105 scenarios"
echo "    T_3:  135 scenarios"
echo "    T_4:  120 scenarios"
echo "    T_5:  150 scenarios"
echo "    T_6:  180 scenarios"
echo "    Log: $LOGFILE"
echo "============================================================"
