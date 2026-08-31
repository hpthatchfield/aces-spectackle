#!/usr/bin/env python
"""
Evaluate a finished MOPRA Count baseline run: metrics + diagnostic figures.

Run from repo root:
  python experiments/MOPRA_Count/plots/plot_eval_run.py \\
    --run-dir experiments/MOPRA_Count/runs/mopra_count_<ts>_<tag>

Writes under <run-dir>/figures/ and <run-dir>/eval_report.json, eval_val_predictions.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_MOPRA = _SCRIPT.parent.parent
_REPO = _MOPRA.parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_SCRIPT.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from plot_style import COL_CLEAN, COL_FAIL, COL_MAE, COL_SPEC, COL_SUCCESS, COL_TRAIN  ### noqa: E402

from spectackle.data.mopra_dataset import make_mopra_loaders  ### noqa: E402
from spectackle.models import CountNet1DDeep  ### noqa: E402
from spectackle.plotting import mae_by_true_k  ### noqa: E402
from spectackle.training import batch_model_input  ### noqa: E402


def _load_model(run_dir: Path, manifest: dict) -> CountNet1DDeep:
    args = manifest["args"]
    model = CountNet1DDeep(width=int(args["width"]), n_blocks=int(args["n_blocks"]))
    state = torch.load(run_dir / "count_net.pt", map_location="cpu")
    model.load_state_dict(state)
    return model


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, Kmax: int) -> np.ndarray:
    cm = np.zeros((Kmax + 1, Kmax + 1), dtype=np.float64)
    for t, p in zip(y_true.astype(int), np.clip(y_pred.astype(int), 0, Kmax)):
        cm[t, p] += 1.0
    return cm


@torch.no_grad()
def _collect_val_predictions(model, val_loader, *, device: str, Kmax: int):
    model.eval()
    y_true, y_pred, y_cont = [], [], []
    specs, specs_clean, comp_v, comp_ok = [], [], [], []
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        K_hat = model(x, mask)
        Kp = torch.clamp(torch.round(K_hat), 0, Kmax).long()
        y_true.append(K.cpu().numpy())
        y_pred.append(Kp.cpu().numpy())
        y_cont.append(K_hat.cpu().numpy())
        specs.append(batch["spec"].numpy())
        specs_clean.append(batch["spec_clean"].numpy())
        comp_v.append(batch["component_v_kms"].numpy())
        comp_ok.append(batch["component_valid"].numpy())
    return (
        np.concatenate(y_true).astype(int),
        np.concatenate(y_pred).astype(int),
        np.concatenate(y_cont).astype(np.float32),
        np.concatenate(specs, axis=0),
        np.concatenate(specs_clean, axis=0),
        np.concatenate(comp_v, axis=0),
        np.concatenate(comp_ok, axis=0),
    )


def _component_vlines(ax, v_row: np.ndarray, valid_row: np.ndarray) -> None:
    for k in range(int(v_row.size)):
        if valid_row[k] < 0.5:
            continue
        vx = float(v_row[k])
        if np.isfinite(vx) and vx != 0.0:
            ax.axvline(vx, color="tab:green", ls=":", lw=1.0, alpha=0.85, zorder=3)


def _plot_learning_curves(history: dict, fig_dir: Path) -> None:
    if not history.get("epoch"):
        return
    ep = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax1.plot(ep, history.get("train_loss_epoch", []), "o-", color=COL_TRAIN, label="train SmoothL1")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color=COL_TRAIN)
    ax1.tick_params(axis="y", labelcolor=COL_TRAIN)
    ax2 = ax1.twinx()
    ax2.plot(ep, history.get("val_K_MAE", []), "s-", color=COL_MAE, label="val K MAE")
    ax2.set_ylabel("val K MAE", color=COL_MAE)
    ax2.tick_params(axis="y", labelcolor=COL_MAE)
    ax1.set_title("Training curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_learning_curves.png", dpi=150)
    plt.close(fig)


def _plot_mae_by_k(mae_k: dict[int, float], Kmax: int, fig_dir: Path) -> None:
    ks = list(range(Kmax + 1))
    vals = [mae_k.get(k, float("nan")) for k in ks]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ks, vals, color=COL_MAE, edgecolor="white")
    ax.set_xlabel("true K")
    ax.set_ylabel("MAE |K_pred - K_true|")
    ax.set_title("Validation MAE by true K")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_mae_by_K.png", dpi=150)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, Kmax: int, fig_dir: Path, *, linear: bool) -> None:
    k_show = Kmax + 1
    sub = cm[:k_show, :k_show]
    vmax = max(1.0, float(sub.max()))
    fig, ax = plt.subplots(figsize=(7, 6))
    if linear:
        im = ax.imshow(sub, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
        note = "linear"
        suffix = "eval_confusion_linear.png"
    else:
        sub_plot = np.ma.masked_less_equal(sub, 0)
        im = ax.imshow(sub_plot, aspect="auto", cmap="viridis", norm=LogNorm(vmin=1.0, vmax=vmax))
        note = "log"
        suffix = "eval_confusion_log.png"
    ax.set_xlabel("K$_{pred}$")
    ax.set_ylabel("K$_{true}$")
    ax.set_xticks(np.arange(k_show))
    ax.set_yticks(np.arange(k_show))
    ax.set_title(f"Confusion matrix")
    plt.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(fig_dir / suffix, dpi=150)
    plt.close(fig)


def _plot_pred_vs_true(y_true, y_pred, y_cont, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, c=COL_SPEC, edgecolors="none")
    lim = max(y_true.max(), y_pred.max(), 1) + 0.5
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("K_true")
    ax.set_ylabel("K_pred (rounded)")
    ax.set_title("Rounded prediction vs truth")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(y_true, y_cont, s=8, alpha=0.35, c=COL_TRAIN, edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("K_true")
    ax.set_ylabel("K_hat (continuous)")
    ax.set_title("Raw regression output vs truth")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_pred_vs_true.png", dpi=150)
    plt.close(fig)


def _plot_residual_by_k(y_true, y_cont, Kmax: int, fig_dir: Path) -> None:
    resid = y_cont - y_true.astype(np.float32)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = [resid[y_true == k] for k in range(Kmax + 1)]
    bp = ax.boxplot(data, positions=list(range(Kmax + 1)), widths=0.6, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor(COL_SPEC)
        box.set_alpha(0.55)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("true K")
    ax.set_ylabel("K_hat - K_true")
    ax.set_title("Continuous residual by true K")
    fig.tight_layout()
    fig.savefig(fig_dir / "eval_residual_by_K.png", dpi=150)
    plt.close(fig)


def _pick_success_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_cont: np.ndarray,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    ### Exact rounded match; prefer low continuous residual (confident successes).
    ok = np.where(y_true == y_pred)[0]
    if ok.size == 0:
        return np.array([], dtype=int)
    resid = np.abs(y_cont[ok] - y_true[ok].astype(np.float32))
    order = np.argsort(resid)
    ok_sorted = ok[order]
    if ok_sorted.size <= n:
        return ok_sorted
    ### Take the best half by residual, then random sample for variety.
    rng = np.random.default_rng(seed)
    pool = ok_sorted[: max(n * 3, n)]
    k = min(n, pool.size)
    return np.sort(rng.choice(pool, size=k, replace=False))


def _pick_failure_indices(y_true: np.ndarray, y_pred: np.ndarray, *, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### Returns (all_failures_sorted, under_count, over_count) indices by |dK| desc.
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    err = np.abs(y_pred[wrong] - y_true[wrong])
    order = np.argsort(-err)
    wrong = wrong[order]
    under = wrong[y_pred[wrong] < y_true[wrong]]
    over = wrong[y_pred[wrong] > y_true[wrong]]
    return wrong[:n], under[: max(1, n // 2)], over[: max(1, n // 2)]


def _plot_spectrum_grid(
    indices: np.ndarray,
    y_true,
    y_pred,
    y_cont,
    specs,
    specs_clean,
    comp_v,
    comp_ok,
    v_axis,
    fig_dir: Path,
    *,
    filename: str,
    title: str,
    success: bool,
    n_col: int = 4,
) -> None:
    if indices.size == 0:
        return
    n_plot = int(indices.size)
    n_row = max(1, (n_plot + n_col - 1) // n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 2.6 * n_row), squeeze=False)
    line_col = COL_SUCCESS if success else COL_FAIL
    for k, i in enumerate(indices):
        ax = axes.flat[k]
        ax.plot(v_axis, specs[i], color=COL_SPEC, lw=0.85, alpha=0.9, label="noisy")
        ax.plot(v_axis, specs_clean[i], color=COL_CLEAN, lw=0.85, alpha=0.75, label="clean")
        _component_vlines(ax, comp_v[i], comp_ok[i])
        delta = int(y_pred[i]) - int(y_true[i])
        sign = f"+{delta}" if delta > 0 else str(delta)
        ax.set_title(
            f"idx={i}  K={y_true[i]}->{y_pred[i]} ({sign})  K_hat={y_cont[i]:.2f}",
            fontsize=8,
            color=line_col,
        )
        ax.set_xlabel("v (km/s)")
    for k in range(n_plot, axes.size):
        axes.flat[k].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title}: green dotted = true component centers", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_success_failure_combined(
    success_idx: np.ndarray,
    fail_idx: np.ndarray,
    under_idx: np.ndarray,
    over_idx: np.ndarray,
    y_true,
    y_pred,
    y_cont,
    specs,
    specs_clean,
    comp_v,
    comp_ok,
    v_axis,
    fig_dir: Path,
    *,
    n_each: int,
) -> None:
    ### Top row: successes; middle: under-count failures; bottom: over-count failures.
    n_col = min(4, n_each)
    rows: list[tuple[str, np.ndarray, bool]] = [
        ("Success (exact K match)", success_idx[:n_col], True),
        ("Under-count (K_pred < K_true)", under_idx[:n_col], False),
        ("Over-count (K_pred > K_true)", over_idx[:n_col], False),
    ]
    active = [(label, idx, ok) for label, idx, ok in rows if idx.size > 0]
    if not active:
        return
    fig, axes = plt.subplots(len(active), n_col, figsize=(3.5 * n_col, 2.8 * len(active)), squeeze=False)
    if len(active) == 1:
        axes = axes.reshape(1, -1)
    for r, (label, idx, is_ok) in enumerate(active):
        for c in range(n_col):
            ax = axes[r, c]
            if c >= idx.size:
                ax.axis("off")
                continue
            i = int(idx[c])
            ax.plot(v_axis, specs[i], color=COL_SPEC, lw=0.85)
            ax.plot(v_axis, specs_clean[i], color=COL_CLEAN, lw=0.85, alpha=0.75)
            _component_vlines(ax, comp_v[i], comp_ok[i])
            delta = int(y_pred[i]) - int(y_true[i])
            ax.set_title(
                f"{label.split('(')[0].strip()}  K={y_true[i]}->{y_pred[i]}  K_hat={y_cont[i]:.2f}",
                fontsize=7,
                color=COL_SUCCESS if is_ok else COL_FAIL,
            )
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("v (km/s)")
    fig.suptitle("Success vs failure spectra (val set)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "eval_success_failure_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _failure_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_cont: np.ndarray) -> dict:
    wrong = y_true != y_pred
    under = wrong & (y_pred < y_true)
    over = wrong & (y_pred > y_true)
    resid = y_cont - y_true.astype(np.float32)
    return {
        "n_val": int(y_true.size),
        "n_correct": int((~wrong).sum()),
        "n_wrong": int(wrong.sum()),
        "n_under_count": int(under.sum()),
        "n_over_count": int(over.sum()),
        "frac_correct": float((~wrong).mean()),
        "mae_under_count_only": float(np.abs(y_pred[under] - y_true[under]).mean()) if under.any() else None,
        "mae_over_count_only": float(np.abs(y_pred[over] - y_true[over]).mean()) if over.any() else None,
        "mean_residual": float(resid.mean()),
        "mean_abs_residual": float(np.abs(resid).mean()),
    }


def _plot_failures(
    y_true,
    y_pred,
    y_cont,
    specs,
    specs_clean,
    comp_v,
    comp_ok,
    v_axis,
    fig_dir: Path,
    *,
    fail_idx: np.ndarray,
) -> None:
    _plot_spectrum_grid(
        fail_idx,
        y_true,
        y_pred,
        y_cont,
        specs,
        specs_clean,
        comp_v,
        comp_ok,
        v_axis,
        fig_dir,
        filename="eval_failure_spectra.png",
        title=f"Worst failures (n={fail_idx.size})",
        success=False,
    )


def _plot_successes(
    y_true,
    y_pred,
    y_cont,
    specs,
    specs_clean,
    comp_v,
    comp_ok,
    v_axis,
    fig_dir: Path,
    *,
    success_idx: np.ndarray,
) -> None:
    _plot_spectrum_grid(
        success_idx,
        y_true,
        y_pred,
        y_cont,
        specs,
        specs_clean,
        comp_v,
        comp_ok,
        v_axis,
        fig_dir,
        filename="eval_success_spectra.png",
        title=f"Successes: exact K match (n={success_idx.size})",
        success=True,
    )


def _plot_typical_per_k(
    y_true,
    y_pred,
    y_cont,
    specs,
    specs_clean,
    comp_v,
    comp_ok,
    v_axis,
    Kmax: int,
    fig_dir: Path,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(Kmax + 1, 1, figsize=(10, 1.8 * (Kmax + 1)), squeeze=False)
    for k in range(Kmax + 1):
        pool = np.where(y_true == k)[0]
        ax = axes[k, 0]
        if pool.size == 0:
            ax.set_title(f"K_true={k}  (no val samples)")
            ax.axis("off")
            continue
        i = int(rng.choice(pool))
        ax.plot(v_axis, specs[i], color=COL_SPEC, lw=0.85)
        ax.plot(v_axis, specs_clean[i], color=COL_CLEAN, lw=0.85, alpha=0.75)
        _component_vlines(ax, comp_v[i], comp_ok[i])
        ax.set_title(
            f"K_true={k}  idx={i}  K_pred={y_pred[i]}  K_hat={y_cont[i]:.2f}",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("K")
    axes[-1, 0].set_xlabel("v (km/s)")
    fig.suptitle(f"Random val exemplar per true K (seed={seed})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_dir / "eval_typical_by_K.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MOPRA Count run evaluation + figures.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-failures", type=int, default=12, help="Max failure spectra to plot.")
    parser.add_argument("--n-successes", type=int, default=12, help="Max success spectra to plot.")
    parser.add_argument("--n-combined", type=int, default=4, help="Spectra per row in success/failure combo figure.")
    parser.add_argument("--typical-seed", type=int, default=42)
    parser.add_argument("--success-seed", type=int, default=42, help="RNG for picking success exemplars.")
    parser.add_argument("--linear-confusion", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    run_args = manifest["args"]
    cfg = deepcopy(manifest["cfg"])
    Kmax = int(run_args.get("Kmax", cfg.get("max_components", 10)))

    _, val_loader = make_mopra_loaders(
        cfg,
        n_train=int(run_args.get("n_train", 10_000)),
        n_val=int(run_args.get("n_val", 2_000)),
        bs_train=int(run_args.get("bs_train", 128)),
        bs_val=int(run_args.get("bs_val", 256)),
        shuffle_seed=int(run_args.get("seed", 42)),
    )

    model = _load_model(run_dir, manifest)
    model.to(args.device)
    y_true, y_pred, y_cont, specs, specs_clean, comp_v, comp_ok = _collect_val_predictions(
        model, val_loader, device=args.device, Kmax=Kmax
    )

    mae_k = mae_by_true_k(y_true, y_pred, Kmax)
    mae_all = float(np.abs(y_pred - y_true).mean())
    acc = float((y_pred == y_true).mean())
    cm = _confusion_matrix(y_true, y_pred, Kmax)

    history = {}
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)

    fail_metrics = _failure_metrics(y_true, y_pred, y_cont)
    success_idx = _pick_success_indices(
        y_true, y_pred, y_cont, n=args.n_successes, seed=args.success_seed
    )
    fail_idx, under_idx, over_idx = _pick_failure_indices(y_true, y_pred, n=args.n_failures)

    report = {
        "run_dir": str(run_dir),
        "Kmax": Kmax,
        "val_K_MAE": mae_all,
        "val_K_acc": acc,
        "mae_by_K": {str(k): round(v, 4) if np.isfinite(v) else None for k, v in mae_k.items()},
        "confusion": cm.astype(int).tolist(),
        **fail_metrics,
        "success_indices": success_idx.astype(int).tolist(),
        "failure_indices": fail_idx.astype(int).tolist(),
        "under_count_indices": under_idx.astype(int).tolist(),
        "over_count_indices": over_idx.astype(int).tolist(),
        "manifest_final_mae": manifest.get("final_val_K_MAE"),
        "history_final_mae": history.get("val_K_MAE", [None])[-1] if history.get("val_K_MAE") else None,
    }

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "eval_val_predictions.npz",
        y_true=y_true.astype(np.int32),
        y_pred=y_pred.astype(np.int32),
        y_cont=y_cont.astype(np.float32),
        spec=specs.astype(np.float32),
        spec_clean=specs_clean.astype(np.float32),
        component_v_kms=comp_v.astype(np.float32),
        component_valid=comp_ok.astype(np.float32),
        n_val=int(y_true.size),
    )

    v_axis = val_loader.dataset.v_axis
    _plot_learning_curves(history, fig_dir)
    _plot_mae_by_k(mae_k, Kmax, fig_dir)
    _plot_confusion(cm, Kmax, fig_dir, linear=False)
    if args.linear_confusion:
        _plot_confusion(cm, Kmax, fig_dir, linear=True)
    _plot_pred_vs_true(y_true, y_pred, y_cont, fig_dir)
    _plot_residual_by_k(y_true, y_cont, Kmax, fig_dir)
    _plot_successes(
        y_true, y_pred, y_cont, specs, specs_clean, comp_v, comp_ok, v_axis, fig_dir, success_idx=success_idx
    )
    _plot_failures(
        y_true, y_pred, y_cont, specs, specs_clean, comp_v, comp_ok, v_axis, fig_dir, fail_idx=fail_idx
    )
    _plot_success_failure_combined(
        success_idx,
        fail_idx,
        under_idx,
        over_idx,
        y_true,
        y_pred,
        y_cont,
        specs,
        specs_clean,
        comp_v,
        comp_ok,
        v_axis,
        fig_dir,
        n_each=args.n_combined,
    )
    _plot_typical_per_k(
        y_true, y_pred, y_cont, specs, specs_clean, comp_v, comp_ok, v_axis, Kmax, fig_dir, seed=args.typical_seed
    )

    print("=== MOPRA Count evaluation ===")
    print(f"Run: {run_dir.name}")
    print(f"n_val={report['n_val']}  Kmax={Kmax}")
    print(f"val_K_MAE={mae_all:.4f}  val_K_acc={acc:.3f}  correct={report['n_correct']}  wrong={report['n_wrong']}")
    print(f"under-count={report['n_under_count']}  over-count={report['n_over_count']}")
    print("MAE by true K:")
    for k in range(Kmax + 1):
        v = mae_k.get(k, float("nan"))
        if np.isfinite(v):
            print(f"  K={k:2d}  MAE={v:.3f}")
    print(f"Wrote {run_dir / 'eval_report.json'}, {run_dir / 'eval_val_predictions.npz'}, figures under {fig_dir}/")


if __name__ == "__main__":
    main()
