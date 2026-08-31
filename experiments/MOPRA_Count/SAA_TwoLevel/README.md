# Two-level SAA experiment (Scouse-style, ML)

Mirrors scousepy stage-1 parent SAA coverage + pixel refinement, without human input at stage 1.

## Pipeline

1. **SAA grid** (automatic): scousepy-style coverage on the CMZ cube (`setup_saa_grid_smooth60.py`).
2. **Stage 1**: train `CountNet1DDeep` on synthetic **SAA-averaged** spectra (`run_stage1_saa.py`).
3. **Stage 2**: train `CountNet1DDeepSaaCond` on synthetic **pixels** with parent SAA spectrum + `K_parent` (`run_stage2_pixel.py`).
4. **Cube inference**: stage1 K on each parent SAA -> stage2 K on pixels in footprint (`run_cube_twolevel.py`).

Stage 2 inputs:
- normalized pixel spectrum (channel 1)
- normalized parent SAA spectrum (channel 2)
- `K_parent` embedding from stage 1

## Quick start

```bash
# 1) SAA grid on smooth60 (no labels)
python experiments/MOPRA_Count/SAA_TwoLevel/setup_saa_grid_smooth60.py

# 2) Stage 1 - parent SAA K model (8 epochs)
python experiments/MOPRA_Count/SAA_TwoLevel/run_stage1_saa.py --epochs 8 --tag saa2_stage1

# 3) Stage 2 - pixel model conditioned on parent (8 epochs)
python experiments/MOPRA_Count/SAA_TwoLevel/run_stage2_pixel.py \\
  --stage1-run-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa2_stage1_<ts>_saa2_stage1 \\
  --epochs 8 --tag saa2_stage2

# 4) Real-cube map + Henshaw compare (Scouse-labeled pixels in SAA footprints)
python experiments/MOPRA_Count/SAA_TwoLevel/run_cube_twolevel.py \\
  --cube data/CMZ_3mm_HNCO_60.fits \\
  --saa-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa_grid_smooth60 \\
  --stage1-run-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa2_stage1_<ts>_saa2_stage1 \\
  --stage2-run-dir experiments/MOPRA_Count/SAA_TwoLevel/runs/saa2_stage2_<ts>_saa2_stage2 \\
  --out data/mopra_cmz_k_pred_saa2.fits \\
  --compare-scouse --infer-on-scouse-labels
```

Figures land in `experiments/MOPRA_Count/figures/` (same stem as `--out`).

## Synthetic training model

Parent spectra = mean of `n_avg` (default 81 ~ 9x9) independent noisy draws from the same synthetic clean profile. Pixel spectra = other draws from the same profile. Labels use `scouse_dat` K (Scouse-accepted component count). Stage 2 teacher-forces `K_parent` from the patch label; at inference `K_parent` comes from stage 1.

## Code locations

| Piece | Path |
|-------|------|
| Synthetic patch generator | `src/spectackle/data/saa_two_level.py` |
| Stage-2 model | `src/spectackle/models/scheme_b_saa.py` |
| Training loop | `train_scheme_b_saa_cond` in `src/spectackle/training.py` |
| SAA grid geometry | `src/spectackle/data/scouse_saa.py` |

## First run (2026-07-09, synth-only `scouse_dat`, 8 epochs each)

| Stage | Run dir |
|-------|---------|
| SAA grid | `runs/saa_grid_smooth60` (1188 SAAs on smooth60) |
| Stage 1 | `runs/saa2_stage1_<ts>_saa2_stage1` |
| Stage 2 | `runs/saa2_stage2_<ts>_saa2_stage2` |

Scouse compare on labeled pixels inside SAA footprints (`n=5198`):

| Metric | SAA two-level | Scheme B repro | Heatmap v1 |
|--------|---------------|----------------|------------|
| Global MAE | 0.806 | 0.782 | 0.734 |
| K=3 MAE | 1.45 | 1.53 | 1.51 |
| K=3 median dK | -2 | -2 | -2 |

K=3 cores improve slightly vs pixel-only Scheme B, but global MAE is still above heatmap because synth-only training does not close the real/synth gap. Next step: stage-2 fine-tune on `scouse_labeled_smooth60.npz` pairs with predicted `K_parent` at inference time.

## Validation

Compare `mopra_cmz_k_pred_saa2_scouse_stats.json` to `scouse_dat` repro and `heatmap_v1`, especially `by_k_true` for K=3/4 and interior edge bins.
