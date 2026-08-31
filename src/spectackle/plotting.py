### plot_example, collect_count_predictions, mae_by_true_k
import numpy as np


def mae_by_true_k(y_true: np.ndarray, y_pred: np.ndarray, Kmax: int) -> dict[int, float]:
    """MAE per true K, dict k: mean |error|."""
    out = {}
    for k in range(Kmax + 1):
        mask = y_true == k
        if mask.sum() == 0:
            out[k] = float("nan")
        else:
            out[k] = float(np.abs(y_pred[mask] - y_true[mask]).mean())
    return out
import torch

from spectackle.training import batch_model_input


def plot_example(ds, idx: int = 0, title=""):
    """Plot one spectrum from a SyntheticSpectraDataset (spec + spec_clean)."""
    import matplotlib.pyplot as plt

    ex = ds[idx]
    v = ds.v_axis
    plt.figure(figsize=(10, 3))
    plt.plot(v, ex["spec"].numpy(), label="spec (noisy)", lw=1.8, alpha=0.8)
    plt.plot(v, ex["spec_clean"].numpy(), label="spec_clean", lw=1.8, alpha=0.8)
    plt.title(title or f"Example index={idx}  K_true={int(ex['K_true'].item())}")
    plt.xlabel("Velocity (km/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def collect_count_predictions_b(
    model, loader, *, device="cpu", Kmax=10, max_batches=50
):
    """Collect (y_true, y_pred) for a Scheme B model (rounded K_hat)."""
    model.eval()
    y_true, y_pred = [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        K_hat = model(x, mask)
        Kp = torch.clamp(torch.round(K_hat), 0, Kmax).long()
        y_true.append(K.cpu().numpy())
        y_pred.append(Kp.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


@torch.no_grad()
def collect_predictions_with_spectra_b(
    model, loader, *, device="cpu", Kmax=10, max_batches=50
):
    """
    Collect (y_true, y_pred, specs, specs_clean) for failure analysis.
    specs, specs_clean: (N, L) numpy arrays.
    """
    model.eval()
    y_true, y_pred, specs, specs_clean = [], [], [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        K_hat = model(x, mask)
        Kp = torch.clamp(torch.round(K_hat), 0, Kmax).long()
        sp = batch["spec"].cpu().numpy()
        sc = batch.get("spec_clean")
        if sc is not None:
            sc = sc.cpu().numpy()
        y_true.append(K.cpu().numpy())
        y_pred.append(Kp.cpu().numpy())
        specs.append(sp)
        if sc is not None:
            specs_clean.append(sc)
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    sp_concat = np.concatenate(specs)
    if specs_clean:
        sc_concat = np.concatenate(specs_clean)
    else:
        sc_concat = None
    if sp_concat.ndim == 3:
        sp_concat = sp_concat.squeeze(1)
    if sc_concat is not None and sc_concat.ndim == 3:
        sc_concat = sc_concat.squeeze(1)
    return yt, yp, sp_concat, sc_concat


@torch.no_grad()
def collect_count_predictions(
    model, loader, *, device="cpu", Kmax=10, max_batches=50
):
    """Collect (y_true, y_pred_argmax, y_pred_expected) for a Scheme C model."""
    model.eval()
    y_true, y_pred, y_exp = [], [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)
        Kp = torch.argmax(logits, dim=1).long()
        ks = torch.arange(Kmax + 1, device=device).float()
        Ke = (probs * ks[None, :]).sum(dim=1)
        y_true.append(K.cpu().numpy())
        y_pred.append(Kp.cpu().numpy())
        y_exp.append(Ke.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_exp)


@torch.no_grad()
def collect_predictions_bc(model_b, model_c, loader, device, Kmax, max_batches=100):
    """Collect predictions from both Scheme B and C for comparison."""
    model_b.eval()
    model_c.eval()
    y_true, y_b, y_c_argmax, y_c_exp = [], [], [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)

        K_b = model_b(x, mask)
        K_b = torch.clamp(torch.round(K_b), 0, Kmax).long()

        logits = model_c(x, mask)
        K_c = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        ks = torch.arange(Kmax + 1, device=device).float()
        K_c_exp = (probs * ks[None, :]).sum(dim=1)

        y_true.append(K.cpu().numpy())
        y_b.append(K_b.cpu().numpy())
        y_c_argmax.append(K_c.cpu().numpy())
        y_c_exp.append(K_c_exp.cpu().numpy())

    return (
        np.concatenate(y_true),
        np.concatenate(y_b),
        np.concatenate(y_c_argmax),
        np.concatenate(y_c_exp),
    )
