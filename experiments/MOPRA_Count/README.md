# MOPRA CMZ: Scheme B component count (Scouse handoff)

Train `CountNet1DDeep` on synthetic MOPRA-axis spectra, run on the real CMZ HNCO cube, and compare `K_pred` to Scouse/Henshaw `K` from `final_fits_updated.dat`.

The metric that matters here is agreement with Scouse pixel K on the smoothed cube, not synthetic val MAE by itself.

## Data files (repo `data/`)

| File | Role |
|------|------|
| `CMZ_3mm_HNCO.fits` | Native MOPRA mosaic (327 ch, dv~1.84 km/s); axis metadata for native training |
| `CMZ_3mm_HNCO_60.fits` | Scouse-input smoothed cube (250 ch, dv=2 km/s); inference target + noise calibration |
| `final_fits_updated.dat` | Scouse/Henshaw per-component fits (~8205 rows; K = row count per (l,b)) |
| `MOPRA_CMZ_jones_2012_header.txt` | FITS header fallback if cube absent |

## Core modules (`src/spectackle/data/`)

| Module | Role |
|--------|------|
| `mopra_header.py` | Parse header / FITS -> `n_channels`, `vrange`, dv |
| `mopra_generator.py` | Synthetic generator (`simple`, `scouse_dat`, `default`, `legacy`, experiment variants) |
| `mopra_scouse_accept.py` | Scouse SNR + deblend filter for synthetic K labels |
| `mopra_scouse_labels.py` | Parse `.dat`, spatial split, labeled spectrum cache |
| `mopra_finetune_dataset.py` | Mixed real + synth loaders for fine-tuning |
| `mopra_preprocess.py` | `BLANK=-1` mask/normalize; Scouse-style SNR |
| `mopra_resample.py` | Velocity resampling when infer cube != training channel count |
| `mopra_dataset.py` | `MOPRASpectraDataset`, `make_mopra_loaders` |
| `scouse_saa.py` | SAA grid geometry + `estimate_spectrum_rms` (ScousePy calc_rms) |
| `scouse_fit_handoff.py` | Multi-Gaussian fit from K_pred -> Henshaw-style .dat |

Model: `CountNet1DDeep` (Scheme B), masked pooling.

## Quick start

From repo root. `run_baseline.py` and `run_finetune_scouse.py` cap BLAS threads internally; export these too if you hit segfaults elsewhere on macOS:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Generator sanity (synth vs smooth60 sigma_rms)
python experiments/MOPRA_Count/sanity_check_generator.py

# Pretrain on smooth60 axis + Scouse-aligned synth labels (default; do not use scouse_dat_relaxed)
python experiments/MOPRA_Count/run_baseline.py \
  --epochs 8 --tag scouse_dat --gen-preset scouse_dat

# Best synth-only run: simple_k6 dials with 20k/4k spectra
python experiments/MOPRA_Count/run_baseline.py \
  --epochs 8 --Kmax 6 --n-train 20000 --n-val 4000 \
  --tag simple_k6_20k --gen-preset simple \
  --noise-calibration-cube data/CMZ_3mm_HNCO_60.fits

# Optional: up-weight high-K training loss (same generator; unlike relaxed K prior)
python experiments/MOPRA_Count/run_baseline.py \
  --epochs 8 --tag scouse_dat_kwt \
  --k-train-weights 1,1,1.5,3,5,1,1,1,1,1,1

# Fine-tune: 50% real Scouse labels + 50% scouse_dat synth
python experiments/MOPRA_Count/build_scouse_cache.py
python experiments/MOPRA_Count/run_finetune_scouse.py \
  --init-run-dir experiments/MOPRA_Count/runs/mopra_count_<timestamp>_scouse_dat \
  --epochs 10 --real-frac 0.5 --tag scouse_ft_v1

# Real-cube K map + Scouse residual (primary eval)
python experiments/MOPRA_Count/run_cube_k_map.py \
  --cube data/CMZ_3mm_HNCO_60.fits \
  --run-dir experiments/MOPRA_Count/runs/mopra_count_<timestamp>_scouse_ft_v1 \
  --infer-on-scouse-labels \
  --out data/mopra_cmz_k_pred_scouse_ft_v1.fits \
  --compare-scouse

# Agree / disagree spectra on real cube
python experiments/MOPRA_Count/plots/plot_cube_discrepancy_spectra.py \
  --k-pred data/mopra_cmz_k_pred_scouse_ft_v1.fits \
  --cube data/CMZ_3mm_HNCO_60.fits

# Synthetic val figures (secondary)
python experiments/MOPRA_Count/plots/plot_eval_run.py \
  --run-dir experiments/MOPRA_Count/runs/mopra_count_<timestamp>_scouse_dat

# ScousePy handoff: K_pred -> multi-Gaussian .dat (scipy fit; Henshaw-comparable)
python experiments/MOPRA_Count/run_scouse_fit_handoff.py \
  --k-pred data/mopra_cmz_k_pred_scouse_ft_v1.fits \
  --cube data/CMZ_3mm_HNCO_60.fits \
  --out data/mopra_cmz_scouse_ft_v1_handoff.dat

python experiments/MOPRA_Count/plots/compare_scouse_fit_handoff.py \
  --pred data/mopra_cmz_scouse_ft_v1_handoff.dat \
  --truth data/final_fits_updated.dat
```

Run artifacts: `experiments/MOPRA_Count/runs/mopra_count_<timestamp>_<tag>/` (`manifest.json`, `history.json`, `count_net.pt`, `curves.png`).

## Heatmap -> K

Center heatmap (Stage 1) then learned K head (Stage 2). Same `simple` dials as the K_reg simple_k6 run.

```bash
python experiments/MOPRA_Count/run_heatmap.py \
  --gen-preset simple --Kmax 6 --n-train 20000 --n-val 4000 --epochs 8 --scheduler \
  --tag heatmap_simple_k6_20k

python experiments/MOPRA_Count/run_heatmap_k.py \
  --heatmap-run-dir experiments/MOPRA_Count/runs/mopra_heatmap_<ts>_heatmap_simple_k6_20k \
  --n-train 20000 --n-val 4000 --epochs 8 --scheduler \
  --tag hm_k_simple_k6_20k

python experiments/MOPRA_Count/run_cube_heatmap_map.py \
  --run-dir experiments/MOPRA_Count/runs/mopra_heatmap_k_<ts>_hm_k_simple_k6_20k \
  --cube data/CMZ_3mm_HNCO_60.fits \
  --out data/mopra_cmz_k_pred_hm_k.fits
```

Optional Scouse fine-tune of the K head: `run_finetune_heatmap_k.py`.

## ScousePy handoff (component table)

Takes an existing `K_pred` FITS map, extracts spectra at Henshaw-labeled pixels, and fits `K = round(K_pred)` Gaussians with scipy (`least_squares`). Writes a 15-column ascii table in the same layout as `final_fits_updated.dat` / ScousePy `output_ascii_indiv`, with `(l,b)` instead of pixel `(x,y)`.

| Piece | Path |
|-------|------|
| Fit + IO | `src/spectackle/data/scouse_fit_handoff.py` |
| Runner | `experiments/MOPRA_Count/run_scouse_fit_handoff.py` |
| Compare vs Henshaw | `experiments/MOPRA_Count/plots/compare_scouse_fit_handoff.py` |

Notes:
- Does not require `pyspeckit` / `scousepy` in the env (MVP uses scipy).
- Width column is Gaussian **dispersion sigma** (km/s), ScousePy convention. Henshaw col7 is treated as **FWHM** in the compare script (`--henshaw-width fwhm`).
- This is a direct pixel handoff, not full ScousePy stages 1-4 (no SAA parent GUI / stage-3 tolerances yet).
- Use `--max-pixels 50` for a quick test.

## Generator presets

| Preset | CLI | Notes |
|--------|-----|-------|
| `scouse_dat` | `--gen-preset scouse_dat` | smooth60 axis (250 ch); K weights from `.dat`; post-draw Scouse acceptance |
| `simple` | `--gen-preset simple` | Fiducial free-sampling (simple_k6): uniform SNR 3-20, FWHM 1.5-60, full-window centers, glance/resolvable-peak K. MOPRA Kmax=6. |
| `simple_residual` | `--gen-preset simple_residual` | Same draws as `simple`; glance SNR>=3 kept; bump-cap = residual flux check (not peak-finder). |
| `simple_matched` | `--gen-preset simple_matched` | Same draws as `simple`; SNR>=3 + 4 km/s merge; credit via matched-filter SNR (true A,W; synth labels only). |
| `heatmap_realamp` | `--gen-preset heatmap_realamp` | Heatmap benchmark: realamp amps/FWHM + clusters; planted centers (no glance/snr/scouse label surgery); Scouse-like K prior. |
| `simple_mix` | `--gen-preset simple_mix` | `simple` plus 50% blend clusters (shoulders / weak secondaries), mild low-K bias. Same glance labels. Experimental vs simple_k6_20k. |
| `simple_realamp` | `--gen-preset simple_realamp` | Ranked amps (primary SNR 4-100) + Henshaw-like lognormal FWHM + mild blend clusters. Glance labels. |
| `simple_realamp_rawk` | `--gen-preset simple_realamp_rawk` | Same morphology as `simple_realamp`, but K = drawn component count (no glance / Scouse filter). |
| `simple_realamp_snrk` | `--gen-preset simple_realamp_snrk` | Same morphology; K = SNR>=3 component count (close blends count; no resolvable-peak cap). Scouse-like K prior. Scouse MAE 0.841 (worse than `simple`/`simple_realamp`; fixes core under-count, inflates edge K=1). |
| `scouse_dat_blend_sat` | `--gen-preset scouse_dat_blend_sat` | FAILED: tight clusters + soft label deblend -> Scouse MAE ~1.97, ~92% over-count. |
| `scouse_dat_relaxed` | `--gen-preset scouse_dat_relaxed` | FAILED ATTEMPT (kept as negative result). Tried to fix bright-center under-count via K up-weight (0.42/0.30/0.18/0.10) + looser blend/amp. Made Scouse match worse (MAE 0.78 -> 1.26). See Findings. |
| `scouse_dat_calibrated` | `--gen-preset scouse_dat_calibrated` | FAILED: always-cluster + amp 0.12 -> Scouse MAE 1.77, K=1 over-count ~93%. |
| `default` | `--gen-preset default` | SNR>=3 draw rules; no post-draw Scouse filter |
| `legacy` | `--gen-preset legacy` | Pre-July 2026 dials (sigma=0.08, SNR 2.5+, more ripple/spike) |

`--noise-calibration-cube data/CMZ_3mm_HNCO_60.fits` re-measures p10-p90 sigma_rms into `noise_std_range` in the manifest.

## Eval

| Stage | Script | Metric |
|-------|--------|--------|
| Synthetic val | `plots/plot_eval_run.py` | MAE on held-out synth |
| Real cube | `run_cube_k_map.py --compare-scouse` | MAE / median dK vs `.dat` |
| Spot check | `plots/plot_cube_discrepancy_spectra.py` | Random agree / large-dK panels |
| Edge vs core | `plots/plot_edge_core_failure_spectra.py` | Edge K=1 over-count vs interior K>=3 under-count (noisy cube) |
| Blend gap | `plots/plot_resolvable_peak_gap.py` | Real vs synth resolvable peaks (`--gen-preset`, `--synth-spec noisy`) |

For Scouse comparison, use `--infer-on-scouse-labels` (infer only at `.dat` positions, SNR gate off). NaN outside Scouse fits; filled where Henshaw published a component.

If the model was trained on native 327 ch but you infer on smooth60, spectra are resampled in velocity. With `scouse_dat` pretrain, axis matches smooth60 and no resample is needed.

Discrepancy plots default to the full cube velocity axis; `--vel-range 40 140` zooms the science window.

## Fine-tune

1. `build_scouse_cache.py`: extract smooth60 spectra at `.dat` (l,b); spatial train/val split (~80/20 by sky cell).
2. `run_finetune_scouse.py`: load `count_net.pt`, mixed real + synth batches.

Fine-tune val is Scouse spatial-val MAE on held-out sky cells. If init checkpoint channel count differs, real cache spectra are velocity-resampled (`--match-init-axis`, default on).

## Findings (July 2026)

Old `biased_low_sep` / native-axis models: synth val MAE ~0.25 but Scouse MAE ~5+ (median K_pred ~7-10 vs ~1-2). Task mismatch: synth counted all drawn Gaussians; Scouse keeps amp/sigma_rms >= 3 and merges blends.

`scouse_dat` pretrain + fine-tune (`scouse_ft_v1`): Scouse MAE ~0.23, median dK 0, K_pred median ~1. Synth val MAE ~0.02 on `scouse_dat` (expected; same generator). That 0.23 is a fine-tuned result; the synth-only table below is the fair non-fine-tuned comparison.

### Recent synth-only `simple` results

All rows use `Kmax=6`, eight epochs, noise calibration from `CMZ_3mm_HNCO_60.fits`, and evaluation on the same 5224 Scouse-labeled pixels. An earlier `simple` run with `Kmax=10` is superseded by `simple_k6` and later.

| Run | Train / val | Generator change | Synth val MAE | Scouse MAE | Exact |
|-----|-------------|------------------|---------------|------------|-------|
| `simple_k6` | 10k / 2k | Uniform SNR 3-20, FWHM 1.5-60, glance peak fraction 0.15 | 0.211 | 0.512 | 0.550 |
| `simple_v2` | 10k / 2k | Log-uniform SNR 3-100, FWHM 4-60, peak fraction 0.25 | 0.196 | 0.794 | 0.407 |
| `simple_v3` | 20k / 4k | As v2, but FWHM floor 2 km/s | 0.121 | 0.753 | 0.428 |
| **`simple_k6_20k`** | **20k / 4k** | **Restored `simple_k6` dials** | 0.172 | **0.480** | **0.582** |
| `simple_k6_30k` | 30k / 6k | Same dials as `simple_k6_20k` | 0.161 | 0.484 | 0.578 |

`simple_k6_20k` is the best synth-only run. Increasing to 30k improved synthetic validation but not real-cube agreement. The 20k gain came mainly from fewer K=1 errors and less over-counting. Crowded K>=3 cores still under-count: `simple_k6_20k` K=3 MAE is 1.29 and the mean dK at 10-20 pixels from an edge is -0.88.

The current `simple` preset is the restored `simple_k6` configuration. `simple_v2` and `simple_v3` are historical runs; their exact configs remain in their run manifests.

Legacy inference note: global peak/RMS SNR passed pixels whose lines sit outside 40-140 km/s; use `--snr-method scouse` if you SNR-gate instead of label-mask.

Failed attempt, `scouse_dat_relaxed` (Jul 2026): tried to reduce the bright-center under-count on the non-fine-tuned model by up-weighting K=3/4 in the draw and loosening deblend/amp. Result: Scouse MAE went 0.78 -> 1.26, K=1 flipped to strong over-count (mean dK +1.47), edge over-count grew (+1.84 at 1-2 px), and bright cores still under-counted (K=3 mean -1.21, K=4 -1.74). Two reasons: (1) the deblend/amp loosening was inert (synthetic K=3/4 keep ~2.84/3.75 resolvable peaks either way; the separation floor rarely binds), so only the K prior actually changed; (2) the model behaves as a resolvable-peak counter, and real crowded cores show ~0.8 resolvable peaks vs ~3 in synthetic K=3, so a higher count prior only inflates low-information (faint/edge/K=1) pixels while the bright cores follow the single-peak evidence. Fixing this needs blend realism calibrated to real component shapes, not a count-prior change.

## D-lite (K + velocity centers)

Same `simple` glance generator as Scheme B, but `CenterNet1DDeep` with an optional velocity-slot head.

```bash
# Match Scheme B K objective (SmoothL1 scalar), no v loss (K parity check)
python experiments/MOPRA_Count/run_dlite.py \
  --epochs 8 --Kmax 6 --n-train 20000 --n-val 4000 \
  --tag dlite_simple_k6_20k_reg_wv0 --gen-preset simple \
  --noise-calibration-cube data/CMZ_3mm_HNCO_60.fits \
  --k-mode reg --no-coord --w-v 0 --scheduler

python experiments/MOPRA_Count/run_cube_dlite_map.py \
  --cube data/CMZ_3mm_HNCO_60.fits \
  --run-dir experiments/MOPRA_Count/runs/mopra_dlite_<ts>_dlite_simple_k6_20k_reg_wv0 \
  --out data/mopra_cmz_k_pred_dlite_simple_k6_20k_reg_wv0.fits \
  --compare-scouse
```

| Run | Scouse MAE | Notes |
|-----|------------|-------|
| Scheme B `simple_k6_20k` | **0.480** | reference |
| D-lite CE + simple + coord | 0.776 | old objective |
| D-lite CE, w_v=0, no coord | 0.628 | still trails B on real cube |
| D-lite **reg**, w_v=0, no coord | **0.517** | near B; current K default |
| D-lite reg + coord, w_v=0.25 | 0.616 | joint v hurts K; need K warmup / freeze |

`--k-mode reg` is required for MOPRA K parity. CE matches B on synth val MAE but not on the real cube. Joint v training at `w_v=0.25` still costs ~0.1 K MAE; next is K-only warmup then freeze encoder for v, or init from Scheme B weights. Velocity QA vs Henshaw is next once K is locked.

## Plot scripts

| Script | Output |
|--------|--------|
| `plot_eval_run.py` | Learning curves, confusion, synth success/failure spectra |
| `plot_k_map_image.py` | K_pred map PNG |
| `plot_k_residual_map.py` | dK vs Scouse + JSON stats |
| `plot_cube_discrepancy_spectra.py` | Real-cube agree/disagree QA |
| `compare_scouse_fit_handoff.py` | Handoff .dat vs Henshaw (K + matched dv/amp/width) |

## Scouse / SAA (exploratory)

| Script | Role |
|--------|------|
| `setup_saa_grid.py` | Stage-1 SAA coverage grid for ScousePy |
| `run_saa_k_predict.py` | K_pred at SAA centres (not primary eval yet) |

## Generator v2 (`default`) vs `legacy`

Target: component acceptance closer to Scouse/Henshaw, noise from smoothed CMZ cube.

| Parameter | Legacy | Default (v2) | Why |
|-----------|--------|--------------|-----|
| `noise_std_range` | fixed 0.08 | 0.012-0.032 K (smooth60 p10-p90) | Real sigma_rms med ~0.021 K |
| `snr_range` | 2.5-18 | 3.0-15 | Scouse component SNR tol |
| `min_peak_height_factor` | 2.0 | 3.0 | Weakest peak >= 3sigma |
| `min_amp_ratio` | 0.25 | 0.40 | Stronger secondaries |
| `min_component_separation` | 2.5 | 3.0 | Stricter sigma-sum spacing |
| `min_sep_channels` | 3 | 5 | ~9.2 km/s floor at native dv |
| `blend_cluster_prob` | 0.08 | 0.04 | Less artificial crowding |
| `ripple_prob` / `spike_prob` | 0.35 / 0.12 | 0.12 / 0.04 | Artifacts inflated K on real data |

Component velocities still draw over the full training axis (Scouse fits negative LSR v too).
