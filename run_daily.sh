#!/bin/bash
# run_daily.sh — Between-survey daily forward pipeline for Little Professor.
#
# Runs every morning after yesterday's station data is available.
# Does NOT resample, recluster, or rewrite SMETs — those happen in
# run_little_prof.sh when a new UAS survey lands.
#
# Sequence:
#   1. aws_ingest.py    — fetch latest station records into parquet cache
#   2. smet_append.py   — append new hourly rows to per-cluster SMETs
#   3. nwp_ingest.py    — truncate + extend SMETs with WRF forecast rows
#   4. SNOWPACK         — incremental advance from restart files
#   5. Save anchor .sno — observation-grounded state for NWP restore
#   6. Restore anchor + SNOWPACK forecast pass (T_stable → T_stable+48h)
#   7. analysis         — extract Sk38/SSI/slab features at today's snapshot
#   8. scenarios        — trigger selection, release polygons, depth rasters
#
# Usage:
#   ./run_daily.sh                        # defaults, today's snapshot
#   ./run_daily.sh --snapshot 2026-02-15  # explicit snapshot date
#   ./run_daily.sh --no-nwp               # skip NWP forecast extension
#   ./run_daily.sh --dry-run              # report without writing SMET rows
#
# Cron:
#   0 8 * * *  /home/ron/avachain/run_daily.sh >> /var/log/avachain_daily.log 2>&1

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLOPE_SCRIPTS=$PROJECT_DIR/slopes/little_prof
SLOPE_DIR=${SLOPE_DIR:-$PROJECT_DIR/snowpack/little_prof}
TOML=$PROJECT_DIR/slopes/little_prof/slope_config.toml
PIPELINE="python $PROJECT_DIR/src/avachain/forcing_pipeline.py --toml $TOML"
ANALYSIS="python $PROJECT_DIR/src/avachain/analysis_pipeline.py --toml $TOML"
AWS="python $PROJECT_DIR/src/avachain/aws_ingest.py --toml $TOML"
SMET_APPEND="python $PROJECT_DIR/src/avachain/smet_append.py --toml $TOML"
NWP="python $PROJECT_DIR/src/avachain/nwp_ingest.py --toml $TOML"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/little_prof/logs

SNAPSHOT_DATE=""
NO_NWP=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --snapshot)   SNAPSHOT_DATE="$2"; shift 2 ;;
        --no-nwp)     NO_NWP=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/daily_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Little Professor — Daily Forward Pipeline"
echo "  Started: $(date)"
echo "  No-NWP:  $NO_NWP"
echo "  Dry-run: $DRY_RUN"
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

# Save each cluster's output .sno as its _anchor.sno.
save_anchor_sno() {
    local n=0
    for res_sno in "$SLOPE_DIR/output"/cluster_*_cluster_*.sno; do
        [[ -f "$res_sno" ]] || continue
        stem=$(basename "$res_sno")
        cid="${stem%%_cluster_*.sno}"
        cp "$res_sno" "$SLOPE_DIR/output/${cid}_anchor.sno"
        n=$((n + 1))
    done
    echo "    Saved anchor .sno for $n clusters"
}

# Restore each _anchor.sno → input/snow/ for the NWP forecast pass.
restore_anchor_sno() {
    local n=0
    for anchor_sno in "$SLOPE_DIR/output"/cluster_*_anchor.sno; do
        [[ -f "$anchor_sno" ]] || continue
        stem=$(basename "$anchor_sno")
        cid="${stem%_anchor.sno}"
        cp "$anchor_sno" "$SLOPE_DIR/input/snow/${cid}.sno"
        n=$((n + 1))
    done
    echo "    Restored anchor .sno for $n clusters → input/snow/"
}

SECONDS=0
SNAPSHOT_ARG=""
[[ -n "$SNAPSHOT_DATE" ]] && SNAPSHOT_ARG="--snapshot-date $SNAPSHOT_DATE"
DRY_ARG=""
[[ $DRY_RUN -eq 1 ]] && DRY_ARG="--dry-run"

echo ">>> Step 1: Fetch latest station data"
step1_start=$SECONDS
$AWS
step1_elapsed=$((SECONDS - step1_start))
echo "    Done: $((step1_elapsed / 60))m $((step1_elapsed % 60))s"
echo ""

echo ">>> Step 2: Append new rows to cluster SMETs"
step2_start=$SECONDS
if [[ $DRY_RUN -eq 1 ]]; then
    $SMET_APPEND --dry-run
else
    $SMET_APPEND
fi
step2_elapsed=$((SECONDS - step2_start))
echo "    Done: $((step2_elapsed / 60))m $((step2_elapsed % 60))s"
echo ""

echo ">>> Step 3: NWP forecast extension"
step3_start=$SECONDS
NWP_EDATE=""
STABLE_TS=""
if [[ $NO_NWP -eq 0 ]]; then
    $NWP --lead-hours 48 $DRY_ARG || echo "    WARNING: nwp_ingest failed — continuing without NWP extension"
    _ts_output=$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/src/avachain')
from nwp_ingest import get_stable_ts
from config import ProjectConfig
from pathlib import Path
import pandas as pd
cfg = ProjectConfig.from_toml(Path('$TOML'))
wrf = Path(cfg.wrf_smet_file) if cfg.wrf_smet_file else None
if wrf and wrf.exists():
    ts = get_stable_ts(wrf)
    if ts:
        print(ts.strftime('%Y-%m-%dT%H:%M'),
              (ts + pd.Timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M'))
" 2>/dev/null)
    if [[ -n "$_ts_output" ]]; then
        STABLE_TS=$(echo "$_ts_output" | cut -d' ' -f1)
        NWP_EDATE=$(echo "$_ts_output" | cut -d' ' -f2)
        echo "    T_stable  = $STABLE_TS"
        echo "    NWP EDATE = $NWP_EDATE (T_stable + 48h)"
    else
        echo "    WRF not available; SNOWPACK will use default season end"
    fi
else
    echo "    Skipped (--no-nwp)"
fi
step3_elapsed=$((SECONDS - step3_start))
echo "    Done: $((step3_elapsed / 60))m $((step3_elapsed % 60))s"
echo ""

echo ">>> Step 4: SNOWPACK (incremental)"
step4_start=$SECONDS
if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$STABLE_TS" && $NO_NWP -eq 0 ]]; then
        echo "    Observed pass: → $STABLE_TS"
        bash "$SLOPE_SCRIPTS/run_snowpack.sh" "" "$STABLE_TS"
        echo ""
        echo "    Saving anchor .sno (observation-grounded state at T_stable)"
        save_anchor_sno
        echo ""
        echo "    Restoring anchor .sno for forecast pass"
        restore_anchor_sno
        echo "    Forecast pass: $STABLE_TS → $NWP_EDATE"
        bash "$SLOPE_SCRIPTS/run_snowpack.sh" "" "$NWP_EDATE"
    else
        echo "    Incremental from restart files (no NWP end date)"
        bash "$SLOPE_SCRIPTS/run_snowpack.sh"
        echo ""
        echo "    Saving anchor .sno"
        save_anchor_sno
    fi
else
    echo "    Dry-run: skipping SNOWPACK"
fi
step4_elapsed=$((SECONDS - step4_start))
echo "    Done: $((step4_elapsed / 60))m $((step4_elapsed % 60))s"
echo ""

echo ">>> Step 5: Analysis + scenario generation"
step5_start=$SECONDS
if [[ $DRY_RUN -eq 0 ]]; then
    $ANALYSIS analyze $SNAPSHOT_ARG
    $ANALYSIS scenarios $SNAPSHOT_ARG
fi
step5_elapsed=$((SECONDS - step5_start))
echo "    Done: $((step5_elapsed / 60))m $((step5_elapsed % 60))s"
echo ""

total_elapsed=$SECONDS
echo "============================================================"
echo "  Daily forward complete"
echo "  Finished: $(date)"
echo "  Total:    $((total_elapsed / 3600))h $(((total_elapsed % 3600) / 60))m $((total_elapsed % 60))s"
echo ""
echo "  Step 1 (aws_ingest):  $((step1_elapsed / 60))m $((step1_elapsed % 60))s"
echo "  Step 2 (smet_append): $((step2_elapsed / 60))m $((step2_elapsed % 60))s"
echo "  Step 3 (nwp_ingest):  $((step3_elapsed / 60))m $((step3_elapsed % 60))s"
echo "  Step 4 (SNOWPACK):    $((step4_elapsed / 60))m $((step4_elapsed % 60))s"
echo "  Step 5 (analysis):    $((step5_elapsed / 60))m $((step5_elapsed % 60))s"
echo ""
echo "  Scenarios: $PROJECT_DIR/outputs/little_prof/scenarios/"
echo "  Log:       $LOGFILE"
echo "============================================================"
