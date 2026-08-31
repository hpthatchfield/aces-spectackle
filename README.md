# SpecTackle

ML workflow for decomposing gaussian spectra, very much still a WIP. Currently, limited scope to figuring out how many Gaussian components are in a 1D spectrum. The idea is to train on synthetic data first, then eventually plug into something like ScousePy that can fit the actual line parameters once it knows how many to look for.

## What's in here

- **K_reg** (old Scheme B): Regression -> predict a scalar K (number of components). SmoothL1 loss.
- **Scheme C**: Classification -> predict a distribution over 0..Kmax. Cross-entropy.
- **Heatmap -> K**: Stage-1 predicts a 1D center heatmap, Stage-2 reads that heatmap and predicts K.

Training is still mostly synthetic Gaussian mixtures + noise + baseline. MOPRA also runs on the real CMZ cube and compares `K_pred` to Scouse/Henshaw K. ACES heatmap is the same idea on the native HNCO velocity grid. Cubes and checkpoints stay local (`data/` and `**/runs/` are gitignored).

## Quick start

```bash
git clone https://github.com/hpthatchfield/aces-spectackle.git
cd ACES_SpecTackle
pip install -e .
```

On macOS, cap BLAS threads before training or you can hit segfaults:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

The current default runs for here / Mina are in `baselines/` (ACES heatmap, MOPRA heatmap, and K_reg). See `baselines/README.md`. You will need the cubes under `data/` yourself; they are not in git.

Longer notes and variants are still under `experiments/MOPRA_Count/` and `experiments/ACES_Heatmap/`. If you just want the original synthetic-only notebook loop, `experiments/SpecTackle_SchemeB.ipynb` still generates data on the fly and trains a small CNN.

## Structure

- `baselines/` - current default runs (ACES / MOPRA heatmap then K, plus K_reg)
- `src/spectackle/` - core package with data generation, models, training, plotting. Probs going to change a lot as we go for the time being.
- `experiments/` - experiment folders + READMEs (`MOPRA_Count`, `ACES_Heatmap`, plus older B/C notebooks)
- `scripts/` - validation scripts (kinda incomplete rn)
- `docs/` - notes (when I get to it)

## Status

WIP. K_reg and Scheme C both train and converge on synthetic data. The thing we actually want to run on Mina is in `baselines/`. Lots to mess around with, lmk if you have any ideas!

## License

MIT. See [LICENSE](LICENSE).
