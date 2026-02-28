#!/usr/bin/env python
### Sanity run: imports + minimal training. Assumes pip install -e .
from spectackle.config import deep_update, set_cpu_safety
from spectackle.data import BASE_CFG, make_loaders
from spectackle.models import CountNet1D, CountNet1D_Classify
from spectackle.training import train_scheme_b, train_scheme_c
from spectackle.plotting import collect_predictions_bc

import torch

cfg = deep_update(BASE_CFG, {})
Kmax = int(cfg["max_components"])
train_loader, val_loader = make_loaders(cfg, n_train=200, n_val=100, bs_train=32, bs_val=32)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_b = CountNet1D(width=32)
model_b = train_scheme_b(model_b, train_loader, val_loader, device=device, epochs=1, log_every=20, Kmax=Kmax)

model_c = CountNet1D_Classify(Kmax=Kmax, width=32)
model_c = train_scheme_c(model_c, train_loader, val_loader, device=device, epochs=1, log_every=20)

y_true, y_b, y_c_argmax, y_c_exp = collect_predictions_bc(model_b, model_c, val_loader, device, Kmax, max_batches=5)
print("Phase 2 OK")
