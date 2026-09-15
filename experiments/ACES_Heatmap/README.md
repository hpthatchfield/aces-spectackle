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

Synth vs real morphology gallery (labeled synth + unlabeled synth|region1):

```bash
$PY experiments/ACES_Heatmap/plots/plot_synth_vs_real_gallery.py --gen-preset simple_snr
$PY experiments/ACES_Heatmap/sanity_check_generator.py
```

`islands` (try this for blended + unusual configs): K is the outcome of a few
velocity families. Each family is a bright primary plus a short extra chain
(secondary / tertiary, then a thin exotic tail). Kmax=10 is a cap, not a target.

```bash
$PY experiments/ACES_Heatmap/sanity_check_generator.py --gen-preset islands --Kmax 10 --n-samples 2000
$PY experiments/ACES_Heatmap/plots/plot_synth_vs_real_gallery.py --gen-preset islands --Kmax 10
```

## Mina Stage 1 (2026-09-01)

First full `simple_snr` heatmap train (20k/4k, 8 epochs, scheduler, kernel 25, +/-80 km/s). Weights stay on Mina; plots + manifest are copied here.

- Mina run: `baselines/runs/aces_heatmap_2026-09-01T204928Z_simple_snr_k6`
- Copy: `experiments/ACES_Heatmap/records/aces_heatmap_2026-09-01T204928Z_simple_snr_k6/` (`curves.png`, `example_heatmaps.png`, `manifest.json`, `history.json`)
- final val: loss 0.1707, peak_prob 0.797, RF 145 ch (~30 km/s)
- peak-decode (side check, not the K model): height 0.25, prom 0.08, K_MAE 0.360, exact 0.724

## Mina Stage 2 (2026-09-02)

K head on the frozen Stage 1 heatmap. Same 20k/4k/8.

- Mina run: `baselines/runs/aces_heatmap_k_2026-09-02T225052Z_simple_snr_k6`
- Copy: `experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-02T225052Z_simple_snr_k6/`
- K head: MAE 0.161, exact 0.845  (peak-decode still 0.360 / 0.724)
- Per-K success/failure gallery (next to the Aug glance one):
  `experiments/ACES_Heatmap/figures/failure_spectra/aces_heatmap_k_2026-09-02T225052Z_simple_snr_k6_val_by_k.png`

Region1 cube map (native ACES cutout, not the 0.25 km/s NLW subcube):
`experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-02T225052Z_simple_snr_k6/hnco_region1_aces_hm_k_pred.fits`
K map: `experiments/ACES_Heatmap/figures/region1/simple_snr_k_map.png`
(the old glance map is `figures/region1/simple_glance_k_map.png`)

## Mina islands Stage 1 (2026-09-09)

Full `islands` heatmap train (20k/4k, 8 epochs, scheduler, kernel 25, +/-80 km/s, Kmax=10). Weights stay on Mina; plots + manifest are copied here.

- Mina run: `experiments/ACES_Heatmap/runs/aces_heatmap_2026-09-09T230054Z_aces_hm_islands_k10_pm80`
- Copy: `experiments/ACES_Heatmap/records/aces_heatmap_2026-09-09T230054Z_aces_hm_islands_k10_pm80/`
- final val: loss 0.290, peak_prob 0.649, RF 145 ch (~30 km/s)
- peak-decode (side check, not the K model): height 0.25, prom 0.08, K_MAE 1.118, exact 0.345

## Mina islands Stage 2 (2026-09-12)

K head on the frozen Stage 1 heatmap. Same 20k/4k/8 as Stage 1.

- Mina run: `experiments/ACES_Heatmap/runs/aces_heatmap_k_2026-09-12T223630Z_aces_hm_k_islands_k10_pm80`
- Copy: `experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-12T223630Z_aces_hm_k_islands_k10_pm80/`
- K head: MAE 0.454, exact 0.609  (peak-decode still 1.118 / 0.345)
- Per-K val gallery (2000 val spectra):
  `experiments/ACES_Heatmap/figures/failure_spectra/aces_heatmap_k_2026-09-12T223630Z_aces_hm_k_islands_k10_pm80_val_by_k.png`

Region1 cube map (native ACES cutout):
`experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-12T223630Z_aces_hm_k_islands_k10_pm80/hnco_region1_aces_hm_k_pred_islands.fits`
K map: `experiments/ACES_Heatmap/figures/region1/islands_k_map.png`
Unlabeled spectra (mixed K): `experiments/ACES_Heatmap/figures/region1/islands_spectra_mixed.png`
Per estimated K: `experiments/ACES_Heatmap/figures/region1/islands_k*.png`
(min K=1, med=5, mean=4.55; no K=0 pixels)

```bash
$PY experiments/ACES_Heatmap/plots/plot_region_by_k.py \
  --k-pred experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-12T223630Z_aces_hm_k_islands_k10_pm80/hnco_region1_aces_hm_k_pred_islands.fits \
  --cube data/hnco_region1_native_aces.fits \
  --tag islands
```

The 2-epoch plumbing check is still under
`experiments/ACES_Heatmap/records/aces_heatmap_k_2026-09-11T175059Z_aces_hm_k_islands_k10_local/`
(K head MAE 0.538, exact 0.534).
