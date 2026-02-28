### plot_example, collect_count_predictions
import numpy as np
import torch

from spectackle.training import _norm


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
        x = _norm(batch["spec"].to(device))
        K = batch["K_true"].to(device).long().squeeze(-1)
        K_hat = model(x)
        Kp = torch.clamp(torch.round(K_hat), 0, Kmax).long()
        y_true.append(K.cpu().numpy())
        y_pred.append(Kp.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


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
        x = _norm(batch["spec"].to(device))
        K = batch["K_true"].to(device).long().squeeze(-1)
        logits = model(x)
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
        x = _norm(batch["spec"].to(device))
        K = batch["K_true"].to(device).long().squeeze(-1)

        K_b = model_b(x)
        K_b = torch.clamp(torch.round(K_b), 0, Kmax).long()

        logits = model_c(x)
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
