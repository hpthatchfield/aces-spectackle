# SpecTackle

ML for estimating how many Gaussian components are in a 1D spectrum. The idea is to train on synthetic data first, then eventually plug into something like ScousePy that can fit the actual line parameters once it knows how many to look for. Right now it's all synthetic spectra and we're just trying to get the count right.

## What's in here

- **Scheme B**: Regression → predict a scalar K (number of components). Uses SmoothL1 loss.
- **Scheme C**: Classification → predict a distribution over 0..Kmax. Uses cross-entropy.
- Both work on 1D spectra (intensity vs velocity) with synthetic Gaussian mixtures + noise + baseline.

We're still on synthetic data only. Real PPV cubes and ScousePy integration are down the road.

## Quick start

```bash
git clone <your-repo-url>
cd ACES_SpecTackle
pip install -e .
```

Then open `experiments/SpecTackle_SchemeB.ipynb` and run the cells. It'll generate data on the fly, train a small CNN, and show some plots. Scheme B typically takes a few minutes on CPU for the default 8 epochs. `SpecTackle_Compare_BC.ipynb` runs both schemes side-by-side if you want to compare.

## Layout

- `src/spectackle/` – core package: data generation, models, training, plotting
- `experiments/` – notebooks (Scheme B, Scheme C, B vs C comparison)
- `scripts/` – validation script for phase 2
- `docs/` – project notes (mostly internal)

## Status

WIP. Scheme B and C both train and converge on synthetic data. Val MAE is usually around 1.1–1.4 for K in 0–10 with the default config. We haven't really pushed on architecture or data difficulty yet – that's next.

## License

MIT. See [LICENSE](LICENSE).
