#!/usr/bin/env bash
### Shared env for baselines/*.sh. Source this; do not run it.

_BASELINES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$_BASELINES_DIR/.." && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_spectackle}"

PY="${PY:-python}"

mkdir -p "$REPO/baselines/runs"

baselines_has_flag() {
    local flag="$1"
    shift
    local a
    for a in "$@"; do
        [[ "$a" == "$flag" ]] && return 0
    done
    return 1
}

baselines_parse() {
    ### Sets BASELINES_QUICK (0/1) and BASELINES_ARGS (remaining flags).
    BASELINES_QUICK=0
    BASELINES_ARGS=()
    local a
    for a in "$@"; do
        if [[ "$a" == "--quick" ]]; then
            BASELINES_QUICK=1
        else
            BASELINES_ARGS+=("$a")
        fi
    done
}

baselines_set_train_flags() {
    ### Default: 20k/4k/8 + scheduler. --quick is a short local check.
    if [[ "${BASELINES_QUICK:-0}" == 1 ]]; then
        TRAIN_FLAGS=(--n-train 2000 --n-val 500 --epochs 2)
    else
        TRAIN_FLAGS=(--n-train 20000 --n-val 4000 --epochs 8 --scheduler)
    fi
}

baselines_utc_ts() {
    date -u +%Y-%m-%dT%H%M%SZ
}
