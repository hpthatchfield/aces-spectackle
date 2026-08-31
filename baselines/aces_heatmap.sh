#!/usr/bin/env bash
### ACES heatmap then K (simple_snr, +/-80 km/s, Kmax=6).
set -eo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

usage() {
    cat <<EOF
ACES heatmap then K (HNCO grid, simple_snr).

  $0 sanity [--quick] [args...]
  $0 stage1 [--quick] [args...]
  $0 stage2 --heatmap-run-dir baselines/runs/<stage1> [--quick] [args...]
  $0 cube --run-dir baselines/runs/<stage2> [--cube /path/to/mosaic] [args...]

Cubes are not in git. Training writes under baselines/runs/.
Set PY if python is not your env.
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
    usage
    exit 0
fi
shift
baselines_parse "$@"
baselines_set_train_flags
ts="$(baselines_utc_ts)"

case "$cmd" in
    sanity)
        exec "$PY" "$REPO/experiments/ACES_Heatmap/sanity_check_generator.py" \
            "${BASELINES_ARGS[@]}"
        ;;
    stage1)
        run_dir="$REPO/baselines/runs/aces_heatmap_${ts}_simple_snr_k6"
        echo "run_dir: $run_dir"
        exec "$PY" "$REPO/experiments/ACES_Heatmap/run_heatmap.py" \
            --gen-preset simple_snr --Kmax 6 \
            --run-dir "$run_dir" \
            "${TRAIN_FLAGS[@]}" \
            "${BASELINES_ARGS[@]}"
        ;;
    stage2)
        if ! baselines_has_flag --heatmap-run-dir "${BASELINES_ARGS[@]}"; then
            echo "stage2 needs --heatmap-run-dir <stage1 run dir>" >&2
            exit 1
        fi
        run_dir="$REPO/baselines/runs/aces_heatmap_k_${ts}_simple_snr_k6"
        echo "run_dir: $run_dir"
        exec "$PY" "$REPO/experiments/ACES_Heatmap/run_heatmap_k.py" \
            --run-dir "$run_dir" \
            "${TRAIN_FLAGS[@]}" \
            "${BASELINES_ARGS[@]}"
        ;;
    cube)
        if ! baselines_has_flag --run-dir "${BASELINES_ARGS[@]}"; then
            echo "cube needs --run-dir <stage2 run dir>" >&2
            exit 1
        fi
        exec "$PY" "$REPO/experiments/ACES_Heatmap/run_cube_heatmap_map.py" \
            --subcube-ref "$REPO/data/hnco_region1_cube.fits" \
            --out "$REPO/data/hnco_region1_aces_hm_k_pred.fits" \
            "${BASELINES_ARGS[@]}"
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
