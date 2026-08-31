# Baselines

Scripts for the current default runs. Same commands here and on Mina.

They call the experiment python with the usual settings filled in. Other experiments stay under `experiments/`.

## What's here

| Script | What | Default settings |
|--------|------|------------------|
| `aces_heatmap.sh` | ACES heatmap, then K | `simple_snr`, +/-80 km/s, Kmax=6 |
| `mopra_heatmap.sh` | MOPRA heatmap, then K | `simple`, Kmax=6, 20k/4k |
| `k_reg.sh` | Predict K as one number (old Scheme B) | MOPRA `simple_k6_20k` |

`k_reg` takes a spectrum and outputs one K. The heatmap scripts first predict a center heatmap, then K.

Outputs go in `baselines/runs/` (not in git). Cubes stay in `data/` (also not in git).

## Env

From repo root. On macOS, cap BLAS threads (the scripts do this too, but set them in the shell if you hit segfaults):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PY=${PY:-python}
```

On Mina, point `PY` at the cluster python that has this env installed.

## ACES heatmap

```bash
./baselines/aces_heatmap.sh sanity
./baselines/aces_heatmap.sh stage1
./baselines/aces_heatmap.sh stage2 --heatmap-run-dir baselines/runs/aces_heatmap_<ts>_simple_snr_k6
./baselines/aces_heatmap.sh cube --run-dir baselines/runs/aces_heatmap_k_<ts>_simple_snr_k6 \
  --cube /path/to/aces_hnco_mosaic.fits
```

`--cube` is the full mosaic. `--subcube-ref data/hnco_region1_cube.fits` is already set (spatial window only).

## MOPRA heatmap

Needs `data/CMZ_3mm_HNCO_60.fits`.

```bash
./baselines/mopra_heatmap.sh sanity
./baselines/mopra_heatmap.sh stage1
./baselines/mopra_heatmap.sh stage2 --heatmap-run-dir baselines/runs/mopra_heatmap_<ts>_simple_k6
./baselines/mopra_heatmap.sh cube --run-dir baselines/runs/mopra_heatmap_k_<ts>_simple_k6
```

## K_reg

Needs `data/CMZ_3mm_HNCO_60.fits`. This is the best synth-only MOPRA count run vs Scouse (MAE ~0.48). Scouse fine-tune is still in `experiments/MOPRA_Count/`.

```bash
./baselines/k_reg.sh sanity
./baselines/k_reg.sh train
./baselines/k_reg.sh cube --run-dir baselines/runs/mopra_k_reg_<ts>_simple_k6
```

## Quick local test

`--quick` uses 2k/500 spectra and 2 epochs. Extra flags after that go through to the python script:

```bash
./baselines/k_reg.sh train --quick --device cpu
```
