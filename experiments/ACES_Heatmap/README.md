# ACES heatmap -> K

Synth-only Stage-1 center heatmap + Stage-2 learned K head on the native ACES HNCO
velocity grid (+/-80 km/s window by default). No Scouse fine-tune here.

## Defaults

| Dial | Value |
|------|-------|
| Preset | `simple_snr` (SNR prune, soft min sep, blend clusters, narrow lognormal FWHM) |
| Alt preset | `simple_glance` (resolvable-peak glance labels; earlier probe) |
| Window | +/-80 km/s at dv~0.208 km/s (~770 ch) |
| Kmax | 6 |
| `label_sigma_kms` | 1.0 |
| `kernel_size` | 25 (wider RF than MOPRA k=9) |
| Decode `min_sep_kms` | 4.0 |
| Target cutout | mosaic + `data/hnco_region1_cube.fits` (data not in git) |

## Run

Use the project env `python` (or set `PY` on the cluster).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_spectackle}
PY=${PY:-python}

# 0) Generator sanity
$PY experiments/ACES_Heatmap/sanity_check_generator.py

# 1) Stage-1 heatmap
$PY experiments/ACES_Heatmap/run_heatmap.py \
  --gen-preset simple_snr \
  --Kmax 6 --n-train 20000 --n-val 4000 --epochs 8 --scheduler \
  --tag aces_hm_simple_snr_k6_pm80

# 2) Stage-2 K head
$PY experiments/ACES_Heatmap/run_heatmap_k.py \
  --heatmap-run-dir experiments/ACES_Heatmap/runs/aces_heatmap_<ts>_aces_hm_simple_snr_k6_pm80 \
  --n-train 20000 --n-val 4000 --epochs 8 --scheduler \
  --tag aces_hm_k_simple_snr_k6_pm80

# 3) region1 cube map (mosaic path via --cube if not under data/)
$PY experiments/ACES_Heatmap/run_cube_heatmap_map.py \
  --run-dir experiments/ACES_Heatmap/runs/aces_heatmap_k_<ts>_aces_hm_k_simple_snr_k6_pm80 \
  --subcube-ref data/hnco_region1_cube.fits \
  --out data/hnco_region1_aces_hm_k_pred.fits
```

For a quick test, add `--n-train 2000 --n-val 500 --epochs 2` on the train scripts.

Synth vs real morphology gallery:

```bash
$PY experiments/ACES_Heatmap/plots/plot_synth_vs_real_gallery.py --gen-preset simple_snr
```
