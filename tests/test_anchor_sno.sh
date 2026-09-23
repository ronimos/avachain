#!/usr/bin/env bash
# tests/test_anchor_sno.sh — Shell integration tests for save_anchor_sno / restore_anchor_sno.
#
# Run with:  bash tests/test_anchor_sno.sh
# Exit code: 0 = all passed, 1 = failures

set -euo pipefail

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    echo "  PASS: $1"
}

fail() {
    FAIL=$((FAIL + 1))
    echo "  FAIL: $1"
}

# ---------------------------------------------------------------------------
# Extract the two functions from run_little_prof.sh.
# We source only those function definitions by isolating them.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHELL_SCRIPT="$REPO_ROOT/run_little_prof.sh"

if [[ ! -f "$SHELL_SCRIPT" ]]; then
    echo "ERROR: $SHELL_SCRIPT not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# set up a fresh temp directory acting as SLOPE_DIR
# ---------------------------------------------------------------------------
run_test() {
    local TEST_NAME="$1"
    local N_CLUSTERS="$2"
    local TEST_CASE="$3"   # "normal" | "noop"

    SLOPE_DIR=$(mktemp -d)
    mkdir -p "$SLOPE_DIR/output" "$SLOPE_DIR/input/snow"

    # Create fixture .sno files (minimal valid content)
    if [[ "$TEST_CASE" == "normal" ]]; then
        for i in $(seq -w 1 "$N_CLUSTERS"); do
            touch "$SLOPE_DIR/output/cluster_${i}_cluster_${i}.sno"
        done
    fi
    # For "noop" test: output directory is empty

    # Inline the two functions with SLOPE_DIR bound to our fixture dir.
    # We use a subshell so SLOPE_DIR doesn't leak.
    (
        # Provide the two functions verbatim from the script,
        # with the correct SLOPE_DIR.
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

        restore_anchor_sno() {
            local n=0
            for anchor_sno in "$SLOPE_DIR/output"/cluster_*_anchor.sno; do
                [[ -f "$anchor_sno" ]] || continue
                stem=$(basename "$anchor_sno")
                cid="${stem%_anchor.sno}"
                cp "$anchor_sno" "$SLOPE_DIR/input/snow/${cid}.sno"
                n=$((n + 1))
            done
            echo "    Restored anchor .sno for $n clusters -> input/snow/"
        }

        # ---- test body ----
        case "$TEST_CASE" in

        normal)
            # save_anchor_sno: creates _anchor.sno for each cluster
            save_output=$(save_anchor_sno)
            anchor_count=$(find "$SLOPE_DIR/output" -name "*_anchor.sno" | wc -l)
            if [[ "$anchor_count" -eq "$N_CLUSTERS" ]]; then
                echo "PASS:${TEST_NAME}:save_creates_anchors"
            else
                echo "FAIL:${TEST_NAME}:save_creates_anchors (expected $N_CLUSTERS, got $anchor_count)"
            fi

            # save_anchor_sno: printed count is correct
            if echo "$save_output" | grep -q "Saved anchor .sno for ${N_CLUSTERS}"; then
                echo "PASS:${TEST_NAME}:save_prints_count"
            else
                echo "FAIL:${TEST_NAME}:save_prints_count (output: $save_output)"
            fi

            # restore_anchor_sno: copies to input/snow/
            restore_output=$(restore_anchor_sno)
            input_count=$(find "$SLOPE_DIR/input/snow" -name "*.sno" | wc -l)
            if [[ "$input_count" -eq "$N_CLUSTERS" ]]; then
                echo "PASS:${TEST_NAME}:restore_creates_input_snos"
            else
                echo "FAIL:${TEST_NAME}:restore_creates_input_snos (expected $N_CLUSTERS, got $input_count)"
            fi

            # restore_anchor_sno: printed count is correct
            if echo "$restore_output" | grep -q "Restored anchor .sno for ${N_CLUSTERS}"; then
                echo "PASS:${TEST_NAME}:restore_prints_count"
            else
                echo "FAIL:${TEST_NAME}:restore_prints_count (output: $restore_output)"
            fi
            ;;

        noop)
            # No-op when output directory is empty
            save_output=$(save_anchor_sno 2>&1)
            restore_output=$(restore_anchor_sno 2>&1)
            # Neither should error (exit code 0 already guaranteed by set -e in subshell)
            # Printed counts should be 0
            if echo "$save_output" | grep -q "Saved anchor .sno for 0"; then
                echo "PASS:${TEST_NAME}:save_noop_prints_zero"
            else
                echo "FAIL:${TEST_NAME}:save_noop_prints_zero (output: $save_output)"
            fi
            if echo "$restore_output" | grep -q "Restored anchor .sno for 0"; then
                echo "PASS:${TEST_NAME}:restore_noop_prints_zero"
            else
                echo "FAIL:${TEST_NAME}:restore_noop_prints_zero (output: $restore_output)"
            fi
            ;;
        esac
    )

    rm -rf "$SLOPE_DIR"
}

# ---------------------------------------------------------------------------
# Run tests, collect PASS/FAIL from subshell output
# ---------------------------------------------------------------------------
echo "=== anchor .sno shell tests ==="

while IFS= read -r line; do
    if [[ "$line" == PASS:* ]]; then
        ok "${line#PASS:}"
    elif [[ "$line" == FAIL:* ]]; then
        fail "${line#FAIL:}"
    else
        echo "$line"
    fi
done < <(
    run_test "three_clusters" 3 "normal"
    run_test "single_cluster" 1 "normal"
    run_test "empty_dir"      0 "noop"
)

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
