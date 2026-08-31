#!/usr/bin/env bash
### MOPRA heatmap then K (simple_k6 dials, Kmax=6).
set -eo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

usage() {
    cat <<EOF
MOPRA heatmap then K (smooth60 axis, gen-preset simple).

  $0 sanity [--quick] [args...]
  $0 stage1 [--quick] [args...]
  $0 stage2 --heatmap-run-dir baselines/runs/<stage1> [--quick] [args...]
  $0 cube --run-dir baselines/runs/<stage2> [args...]

Needs data/CMZ_3mm_HNCO_60.fits for noise calibration and cube maps.
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
        exec "$PY" "$REPO/experiments/MOPRA_Count/sanity_check_generator.py" \
            "${BASELINES_ARGS[@]}"
        ;;
    stage1)
        run_dir="$REPO/baselines/runs/mopra_heatmap_${ts}_simple_k6"
        echo "run_dir: $run_dir"
        exec "$PY" "$REPO/experiments/MOPRA_Count/run_heatmap.py" \
            --gen-preset simple --Kmax 6 \
            --noise-calibration-cube "$REPO/data/CMZ_3mm_HNCO_60.fits" \
            --run-dir "$run_dir" \
            "${TRAIN_FLAGS[@]}" \
            "${BASELINES_ARGS[@]}"
        ;;
    stage2)
        if ! baselines_has_flag --heatmap-run-dir "${BASELINES_ARGS[@]}"; then
            echo "stage2 needs --heatmap-run-dir <stage1 run dir>" >&2
            exit 1
        fi
        run_dir="$REPO/baselines/runs/mopra_heatmap_k_${ts}_simple_k6"
        echo "run_dir: $run_dir"
        exec "$PY" "$REPO/experiments/MOPRA_Count/run_heatmap_k.py" \
            --run-dir "$run_dir" \
            "${TRAIN_FLAGS[@]}" \
            "${BASELINES_ARGS[@]}"
        ;;
    cube)
        if ! baselines_has_flag --run-dir "${BASELINES_ARGS[@]}"; then
            echo "cube needs --run-dir <stage2 run dir>" >&2
            exit 1
        fi
        exec "$PY" "$REPO/experiments/MOPRA_Count/run_cube_heatmap_map.py" \
            --cube "$REPO/data/CMZ_3mm_HNCO_60.fits" \
            --out "$REPO/data/mopra_cmz_k_pred_hm_k.fits" \
            --infer-on-scouse-labels \
            --compare-scouse \
            "${BASELINES_ARGS[@]}"
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
