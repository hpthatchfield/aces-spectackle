#!/usr/bin/env bash
### K_reg: predict K as one number (used to be called Scheme B).
### Default settings match the best synth-only MOPRA run: simple_k6_20k.
set -eo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

usage() {
    cat <<EOF
K_reg: predict K as one number. MOPRA simple_k6_20k settings.

  $0 sanity [--quick] [args...]
  $0 train [--quick] [args...]
  $0 cube --run-dir baselines/runs/<train> [args...]

This is the old Scheme B count model, not heatmap then K.
Needs data/CMZ_3mm_HNCO_60.fits. Fine-tune is still in experiments/MOPRA_Count/.
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
ts="$(baselines_utc_ts)"

### simple_k6_20k did not use --scheduler.
if [[ "$BASELINES_QUICK" == 1 ]]; then
    TRAIN_FLAGS=(--n-train 2000 --n-val 500 --epochs 2)
else
    TRAIN_FLAGS=(--n-train 20000 --n-val 4000 --epochs 8)
fi

case "$cmd" in
    sanity)
        exec "$PY" "$REPO/experiments/MOPRA_Count/sanity_check_generator.py" \
            "${BASELINES_ARGS[@]}"
        ;;
    train)
        run_dir="$REPO/baselines/runs/mopra_k_reg_${ts}_simple_k6"
        echo "run_dir: $run_dir"
        exec "$PY" "$REPO/experiments/MOPRA_Count/run_baseline.py" \
            --gen-preset simple --Kmax 6 \
            --noise-calibration-cube "$REPO/data/CMZ_3mm_HNCO_60.fits" \
            --run-dir "$run_dir" \
            "${TRAIN_FLAGS[@]}" \
            "${BASELINES_ARGS[@]}"
        ;;
    cube)
        if ! baselines_has_flag --run-dir "${BASELINES_ARGS[@]}"; then
            echo "cube needs --run-dir <train run dir>" >&2
            exit 1
        fi
        exec "$PY" "$REPO/experiments/MOPRA_Count/run_cube_k_map.py" \
            --cube "$REPO/data/CMZ_3mm_HNCO_60.fits" \
            --out "$REPO/data/mopra_cmz_k_pred_k_reg.fits" \
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
