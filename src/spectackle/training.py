### train_scheme_b, train_scheme_c (Spectrum normalization + training loops)
import numpy as np
import torch
import torch.nn as nn


def _norm(x):
    """Per-spectrum normalization: subtract mean, divide by std (all channels)."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / (x.std(dim=1, keepdim=True) + 1e-6)


def batch_model_input(batch, device):
    """
    Model input from a loader batch.
    Prefers spec_norm + valid_mask (valid-only normalize, shared with real-cube inference).
    Falls back to _norm(spec) when those keys are absent (legacy loaders).
    """
    if "spec_norm" in batch:
        x = batch["spec_norm"].to(device)
        mask = batch.get("valid_mask")
        return x, mask.to(device) if mask is not None else None
    x = _norm(batch["spec"].to(device))
    return x, None


def train_scheme_b(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=3,
    log_every=200,
    Kmax=10,
    grad_clip=None,
    use_scheduler=False,
    k_loss_weights: np.ndarray | None = None,
    history: dict | None = None,
):
    """Scheme B: regression -> scalar K, SmoothL1 loss.

    Optional k_loss_weights: length-(Kmax+1) array indexed by K_true; up-weights rare
    high-K training examples without changing the synthetic draw (unlike generator K prior).
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")
    w_k: torch.Tensor | None = None
    if k_loss_weights is not None:
        w_k = torch.as_tensor(np.asarray(k_loss_weights, dtype=np.float32), device=device)
        if w_k.numel() != int(Kmax) + 1:
            raise ValueError(f"k_loss_weights must have length Kmax+1={int(Kmax)+1}, got {w_k.numel()}")
    n_steps = len(train_loader)
    w_note = f" k_weights={k_loss_weights.tolist()}" if k_loss_weights is not None else ""
    print(f"[B] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}{w_note}", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_loss = 0.0
        n_loss = 0
        print(f"[B] Epoch {ep}/{epochs}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            x, mask = batch_model_input(batch, device)
            K = batch["K_true"].to(device).float().squeeze(-1)
            K_hat = model(x, mask)
            per = loss_fn(K_hat, K)
            if w_k is not None:
                loss = (per * w_k[K.long().clamp(0, int(Kmax))]).mean()
            else:
                loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            li = float(loss.item())
            running += li
            sum_loss += li
            n_loss += 1
            if step % log_every == 0:
                print(f"  [B]   step {step}/{n_steps}  train_loss {running/log_every:.4f}", flush=True)
                running = 0.0
        model.eval()
        mae, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, mask = batch_model_input(batch, device)
                K = batch["K_true"].to(device).long().squeeze(-1)
                K_hat = model(x, mask)
                K_pred = torch.clamp(torch.round(K_hat), 0, Kmax).long()
                mae += (K_pred - K).abs().sum().item()
                n += x.size(0)
        val_mae = float(mae / max(1, n))
        print(f"  [B]   val_K_MAE {val_mae:.3f}", flush=True)
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_loss / max(1, n_loss)))
            history.setdefault("val_K_MAE", []).append(val_mae)
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


def train_scheme_b_saa_cond(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=3,
    log_every=200,
    Kmax=10,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """Stage-2 SAA-conditioned Scheme B: pixel + parent spec + K_parent embedding."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    n_steps = len(train_loader)
    print(f"[B+SAA] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_loss = 0.0
        n_loss = 0
        print(f"[B+SAA] Epoch {ep}/{epochs}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            x, mask = batch_model_input(batch, device)
            parent = batch["parent_spec_norm"].to(device)
            k_parent = batch["K_parent"].to(device).long().squeeze(-1)
            K = batch["K_true"].to(device).float().squeeze(-1)
            K_hat = model(x, parent, k_parent, mask)
            loss = loss_fn(K_hat, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            li = float(loss.item())
            running += li
            sum_loss += li
            n_loss += 1
            if step % log_every == 0:
                print(f"  [B+SAA]   step {step}/{n_steps}  train_loss {running/log_every:.4f}", flush=True)
                running = 0.0
        model.eval()
        mae, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, mask = batch_model_input(batch, device)
                parent = batch["parent_spec_norm"].to(device)
                k_parent = batch["K_parent"].to(device).long().squeeze(-1)
                K = batch["K_true"].to(device).long().squeeze(-1)
                K_hat = model(x, parent, k_parent, mask)
                K_pred = torch.clamp(torch.round(K_hat), 0, Kmax).long()
                mae += (K_pred - K).abs().sum().item()
                n += x.size(0)
        val_mae = float(mae / max(1, n))
        print(f"  [B+SAA]   val_K_MAE {val_mae:.3f}", flush=True)
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_loss / max(1, n_loss)))
            history.setdefault("val_K_MAE", []).append(val_mae)
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


def train_scheme_c(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=3,
    log_every=200,
    grad_clip=None,
    use_scheduler=False,
):
    """Scheme C: classification -> logits 0..Kmax, CrossEntropy loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_fn = nn.CrossEntropyLoss()
    n_steps = len(train_loader)
    print(f"[C] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        print(f"[C] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            x, mask = batch_model_input(batch, device)
            K = batch["K_true"].to(device).long().squeeze(-1)
            logits = model(x, mask)
            loss = loss_fn(logits, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            if step % log_every == 0:
                print(f"  [C]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        model.eval()
        n, n_correct, mae_argmax, mae_expected = 0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, mask = batch_model_input(batch, device)
                K = batch["K_true"].to(device).long().squeeze(-1)
                logits = model(x, mask)
                Kp = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                ks = torch.arange(probs.shape[1], device=device).float()
                Ke = (probs * ks[None, :]).sum(dim=1)
                n += x.size(0)
                n_correct += (Kp == K).sum().item()
                mae_argmax += (Kp - K).abs().sum().item()
                mae_expected += (Ke - K.float()).abs().sum().item()
        acc = n_correct / max(1, n)
        print(
            f"  [C]   val_acc {acc:.3f}  val_K_MAE(argmax) {mae_argmax/n:.3f}  "
            f"val_K_MAE(E[K]) {mae_expected/n:.3f}"
        )
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_scheme_c_metrics(model, val_loader, *, device="cpu"):
    """
    Validation metrics for Scheme C (Tier 0 parity / reporting).
    Returns mean NLL (natural log), perplexity, accuracy, MAE(argmax), MAE(E[K]), MAE-by-true-K.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    Kmax = None
    n = 0
    total_ce = 0.0
    n_correct = 0
    sum_mae_argmax = 0.0
    sum_mae_exp = 0.0
    all_k, y_pred_a, y_pred_e = [], [], []
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        logits = model(x, mask)
        Kmax = logits.shape[1] - 1
        bs = x.size(0)
        n += bs
        total_ce += loss_fn(logits, K).item()
        Kp = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        ks = torch.arange(probs.shape[1], device=device, dtype=torch.float32)
        Ke = (probs * ks.unsqueeze(0)).sum(dim=1)
        n_correct += (Kp == K).sum().item()
        sum_mae_argmax += (Kp - K).abs().sum().item()
        sum_mae_exp += (Ke - K.float()).abs().sum().item()
        all_k.append(K.cpu().numpy())
        y_pred_a.append(Kp.cpu().numpy())
        y_pred_e.append(Ke.cpu().numpy())
    if n == 0 or Kmax is None:
        raise ValueError("eval_scheme_c_metrics: empty val_loader")
    y_true = np.concatenate(all_k).astype(np.int64)
    pred_a = np.concatenate(y_pred_a).astype(np.int64)
    pred_e = np.concatenate(y_pred_e)

    def _mae_by_true_k(y_t, y_p, km):
        out = {}
        for k in range(km + 1):
            mask = y_t == k
            out[k] = float(np.abs(y_p[mask] - k).mean()) if mask.any() else float("nan")
        return out

    mae_k_a = _mae_by_true_k(y_true, pred_a, Kmax)
    mae_k_e = _mae_by_true_k(y_true, pred_e, Kmax)
    mean_nll = total_ce / n
    return {
        "val_nll": float(mean_nll),
        "val_perplexity": float(np.exp(mean_nll)),
        "val_acc": n_correct / n,
        "val_mae_argmax": sum_mae_argmax / n,
        "val_mae_expected_k": sum_mae_exp / n,
        "n_val": n,
        "Kmax": Kmax,
        "mae_by_true_k_argmax": {str(k): round(v, 4) for k, v in mae_k_a.items()},
        "mae_by_true_k_expected": {str(k): round(v, 4) for k, v in mae_k_e.items()},
    }


@torch.no_grad()
def collect_nlw_predictions(model, val_loader, *, device: str = "cpu"):
    """
    Run validation set; return arrays:
      y_true (N,), logits (N,), spec (N,C), spec_clean (N,C),
      component_v_kms (N,Kmax), component_is_narrow (N,Kmax), component_valid (N,Kmax).
    """
    model.eval()
    ys, logits_l, specs, specs_clean = [], [], [], []
    cvs, cns, coks = [], [], []
    for batch in val_loader:
        x = batch["spec_norm"].to(device)
        mask = batch["valid_mask"].to(device)
        y = batch["y_nlw"].to(device).float().squeeze(-1)
        logit = model(x, mask)
        ys.append(y.cpu().numpy())
        logits_l.append(logit.cpu().numpy())
        specs.append(batch["spec"].numpy())
        specs_clean.append(batch["spec_clean"].numpy())
        cvs.append(batch["component_v_kms"].numpy())
        cns.append(batch["component_is_narrow"].numpy())
        coks.append(batch["component_valid"].numpy())
    y_true = np.concatenate(ys).astype(np.float32)
    logits = np.concatenate(logits_l).astype(np.float32)
    spec = np.concatenate(specs, axis=0)
    spec_clean = np.concatenate(specs_clean, axis=0)
    component_v_kms = np.concatenate(cvs, axis=0)
    component_is_narrow = np.concatenate(cns, axis=0)
    component_valid = np.concatenate(coks, axis=0)
    return y_true, logits, spec, spec_clean, component_v_kms, component_is_narrow, component_valid


def binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr, tpr) with (0,0) prepended; y_score = logits or any monotone score."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return fpr, tpr


def binary_confusion_at_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = (np.asarray(prob, dtype=np.float64).ravel() >= threshold).astype(np.int64)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def tpr_at_max_fpr(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    """Max TPR among ROC points with FPR <= max_fpr."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    mask = fpr <= float(max_fpr) + 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def eval_nlw_extended(y_true: np.ndarray, logits: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Rich metrics from cached val predictions (no model forward)."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    logit = np.asarray(logits, dtype=np.float64).ravel()
    prob = 1.0 / (1.0 + np.exp(-logit))
    cm = binary_confusion_at_threshold(y, prob, threshold=threshold)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    sens = cm["tp"] / max(1, n_pos)
    spec = cm["tn"] / max(1, n_neg)
    prec = cm["tp"] / max(1, cm["tp"] + cm["fp"])
    fpr_rate = cm["fp"] / max(1, n_neg)
    fnr_rate = cm["fn"] / max(1, n_pos)
    fpr_curve, tpr_curve = binary_roc_curve(y, logit)
    auc = _binary_roc_auc(y, logit)
    loss_fn = nn.BCEWithLogitsLoss()
    y_t = torch.from_numpy(y.astype(np.float32))
    logit_t = torch.from_numpy(logit.astype(np.float32))
    val_loss = float(loss_fn(logit_t, y_t).item())
    acc = (cm["tp"] + cm["tn"]) / max(1, int(y.size))
    bal_acc = 0.5 * (sens + spec)
    always_neg_acc = n_neg / max(1, int(y.size))
    return {
        "val_loss": val_loss,
        "val_acc": float(acc),
        "val_auc": float(auc) if np.isfinite(auc) else None,
        "balanced_acc": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "fpr_at_threshold": float(fpr_rate),
        "fnr_at_threshold": float(fnr_rate),
        "tpr_at_fpr_0.01": float(tpr_at_max_fpr(fpr_curve, tpr_curve, 0.01)),
        "tpr_at_fpr_0.005": float(tpr_at_max_fpr(fpr_curve, tpr_curve, 0.005)),
        "threshold": float(threshold),
        "confusion": cm,
        "n_val": int(y.size),
        "n_pos_val": n_pos,
        "n_neg_val": n_neg,
        "always_negative_acc_baseline": float(always_neg_acc),
        "roc_fpr": fpr_curve.tolist(),
        "roc_tpr": tpr_curve.tolist(),
    }


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    ### Mann-Whitney U / midrank AUC; no sklearn. Returns nan if only one class present.
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n = y.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(n, dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        mid = 0.5 * (i + j + 1)
        for k in range(i, j):
            ranks[order[k]] = mid
        i = j
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


@torch.no_grad()
def eval_nlw_metrics(model, val_loader, *, device: str = "cpu", loss_fn=None):
    """
    Validation metrics for NLW binary task.
    Returns dict with val_loss (mean BCE), val_acc (threshold 0.5 on sigmoid), val_auc, n_val, n_pos.
    """
    model.eval()
    if loss_fn is None:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    n_correct = 0
    all_y: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    for batch in val_loader:
        x = batch["spec_norm"].to(device)
        mask = batch["valid_mask"].to(device)
        y = batch["y_nlw"].to(device).float().squeeze(-1)
        logits = model(x, mask)
        bs = x.size(0)
        total_loss += float(loss_fn(logits, y).item()) * bs
        n += bs
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        n_correct += int((pred == y).sum().item())
        all_y.append(y.cpu().numpy())
        all_score.append(logits.cpu().numpy())
    y_cat = np.concatenate(all_y)
    s_cat = np.concatenate(all_score)
    auc = _binary_roc_auc(y_cat, s_cat)
    return {
        "val_loss": float(total_loss / max(1, n)),
        "val_acc": float(n_correct / max(1, n)),
        "val_auc": float(auc) if np.isfinite(auc) else None,
        "n_val": int(n),
        "n_pos_val": int(y_cat.sum()),
    }


def train_nlw_bce(
    model,
    train_loader,
    val_loader,
    *,
    device: str = "cpu",
    lr: float = 1e-3,
    epochs: int = 5,
    log_every: int = 200,
    grad_clip: float | None = None,
    use_scheduler: bool = False,
    pos_weight: torch.Tensor | None = None,
    history: dict | None = None,
):
    """
    Binary NLW presence: BCEWithLogitsLoss on y_nlw in {0,1}.
    Pass pos_weight (scalar tensor) ~ n_neg/n_pos for class imbalance if desired.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    pw = pos_weight.to(device) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    loss_fn_eval = torch.nn.BCEWithLogitsLoss()
    n_steps = len(train_loader)
    print(f"[NLW] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_loss = 0.0
        n_loss = 0
        print(f"[NLW] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            x = batch["spec_norm"].to(device)
            mask = batch["valid_mask"].to(device)
            y = batch["y_nlw"].to(device).float().squeeze(-1)
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            li = float(loss.item())
            running += li
            sum_loss += li
            n_loss += 1
            if step % log_every == 0:
                print(f"  [NLW]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        metrics = eval_nlw_metrics(model, val_loader, device=device, loss_fn=loss_fn_eval)
        auc_s = f"{metrics['val_auc']:.4f}" if metrics["val_auc"] is not None else "nan"
        print(
            f"  [NLW]   val_loss {metrics['val_loss']:.4f}  val_acc {metrics['val_acc']:.3f}  val_auc {auc_s}"
        )
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_loss / max(1, n_loss)))
            history.setdefault("val_loss", []).append(metrics["val_loss"])
            history.setdefault("val_acc", []).append(metrics["val_acc"])
            history.setdefault("val_auc", []).append(metrics["val_auc"])
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


def _scheme_d_oracle_batch_loss(
    model,
    batch,
    *,
    device,
    v_axis: torch.Tensor,
    loss_slot: nn.Module,
    loss_recon: nn.Module,
    w_recon: float,
    w_log_sig: float = 1.0,
    amp_loss_clip: float = 50.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    from spectackle.models.scheme_d import synthesize_gaussian_stack

    x, mask = batch_model_input(batch, device)
    pred = model(x, mask)
    v_pred = pred[:, :, 0]
    log_sig_pred = pred[:, :, 1]
    amp_pred = pred[:, :, 2]
    sigma_pred = torch.exp(log_sig_pred).clamp(min=1e-3, max=500.0)

    v_true = batch["component_v_kms"].to(device)
    log_sig_true = batch["component_log_sigma"].to(device)
    amp_true = batch["component_amp_norm"].to(device)
    slot_ok = batch["component_valid"].to(device)

    n_slot = slot_ok.sum().clamp(min=1.0)
    slot_v = loss_slot(v_pred, v_true)
    ### Relative log-sigma loss (log ratio); up-weighted via w_log_sig.
    slot_sig = loss_slot(log_sig_pred, log_sig_true)
    slot_amp = loss_slot(amp_pred, amp_true)
    amp_ok = slot_ok * (amp_true.abs() <= float(amp_loss_clip)).to(slot_ok.dtype)
    n_amp = amp_ok.sum().clamp(min=1.0)
    slot_loss = (
        (slot_v * slot_ok).sum() / n_slot
        + float(w_log_sig) * (slot_sig * slot_ok).sum() / n_slot
        + (slot_amp * amp_ok).sum() / n_amp
    )

    recon_pred = synthesize_gaussian_stack(
        v_axis, v_pred, sigma_pred, amp_pred, slot_mask=slot_ok
    )
    recon_true = synthesize_gaussian_stack(
        v_axis,
        v_true,
        torch.exp(log_sig_true).clamp(min=1e-3),
        amp_true,
        slot_mask=slot_ok,
    )
    m = batch["valid_mask"].to(device)
    diff = (recon_pred - recon_true).abs() * m
    recon_loss = diff.sum() / m.sum().clamp(min=1.0)

    total = slot_loss + float(w_recon) * recon_loss
    parts = {
        "slot": float(slot_loss.item()),
        "recon": float(recon_loss.item()),
        "total": float(total.item()),
    }
    return total, parts


def train_scheme_d_oracle(
    model,
    train_loader,
    val_loader,
    v_axis: torch.Tensor,
    *,
    device="cpu",
    lr=1e-3,
    epochs=8,
    log_every=200,
    w_recon=0.5,
    w_log_sig=3.0,
    amp_loss_clip=50.0,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """Scheme D Phase 1: oracle-K slot regression + optional recon loss."""
    model.to(device)
    v_axis = v_axis.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_slot = nn.SmoothL1Loss(reduction="none")
    loss_recon = nn.SmoothL1Loss(reduction="none")
    n_steps = len(train_loader)
    print(
        f"[D-oracle] Starting: {epochs} epochs, {n_steps} steps/epoch, "
        f"w_recon={w_recon}  w_log_sig={w_log_sig}"
    )
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_total = 0.0
        n_loss = 0
        print(f"[D-oracle] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            loss, parts = _scheme_d_oracle_batch_loss(
                model,
                batch,
                device=device,
                v_axis=v_axis,
                loss_slot=loss_slot,
                loss_recon=loss_recon,
                w_recon=w_recon,
                w_log_sig=w_log_sig,
                amp_loss_clip=amp_loss_clip,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += parts["total"]
            sum_total += parts["total"]
            n_loss += 1
            if step % log_every == 0:
                print(f"  [D-oracle]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        metrics = eval_scheme_d_oracle_metrics(
            model, val_loader, v_axis=v_axis, device=device, w_recon=w_recon
        )
        print(
            f"  [D-oracle]   val_slot_mae_v {metrics['mae_v']:.3f}  "
            f"mae_log_sig {metrics['mae_log_sig']:.3f}  mae_amp {metrics['mae_amp']:.3f}  "
            f"recon_mae {metrics['recon_mae']:.4f}"
        )
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_total / max(1, n_loss)))
            history.setdefault("val_mae_v", []).append(metrics["mae_v"])
            history.setdefault("val_mae_log_sig", []).append(metrics["mae_log_sig"])
            history.setdefault("val_mae_amp", []).append(metrics["mae_amp"])
            history.setdefault("val_recon_mae", []).append(metrics["recon_mae"])
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_scheme_d_oracle_metrics(
    model,
    val_loader,
    v_axis: torch.Tensor,
    *,
    device="cpu",
    w_recon=0.5,
) -> dict[str, float]:
    from spectackle.models.scheme_d import synthesize_gaussian_stack

    model.eval()
    v_axis = v_axis.to(device)
    sum_v = sum_sig = sum_amp = sum_recon = 0.0
    n_slot = 0.0
    n_recon = 0.0
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        pred = model(x, mask)
        v_pred = pred[:, :, 0]
        log_sig_pred = pred[:, :, 1]
        amp_pred = pred[:, :, 2]

        v_true = batch["component_v_kms"].to(device)
        log_sig_true = batch["component_log_sigma"].to(device)
        amp_true = batch["component_amp_norm"].to(device)
        slot_ok = batch["component_valid"].to(device)

        n_slot += float(slot_ok.sum().item())
        sum_v += ((v_pred - v_true).abs() * slot_ok).sum().item()
        sum_sig += ((log_sig_pred - log_sig_true).abs() * slot_ok).sum().item()
        sum_amp += ((amp_pred - amp_true).abs() * slot_ok).sum().item()

        sigma_pred = torch.exp(log_sig_pred).clamp(min=1e-3, max=500.0)
        recon_pred = synthesize_gaussian_stack(
            v_axis, v_pred, sigma_pred, amp_pred, slot_mask=slot_ok
        )
        recon_true = synthesize_gaussian_stack(
            v_axis,
            v_true,
            torch.exp(log_sig_true).clamp(min=1e-3),
            amp_true,
            slot_mask=slot_ok,
        )
        m = batch["valid_mask"].to(device)
        bs = x.size(0)
        n_recon += float(m.sum().item())
        sum_recon += ((recon_pred - recon_true).abs() * m).sum().item()

    denom = max(1.0, n_slot)
    return {
        "mae_v": float(sum_v / denom),
        "mae_log_sig": float(sum_sig / denom),
        "mae_amp": float(sum_amp / denom),
        "recon_mae": float(sum_recon / max(1.0, n_recon)),
        "w_recon": float(w_recon),
    }


def _scheme_d_lite_oracle_batch_loss(
    model,
    batch,
    *,
    device,
    loss_v: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, mask = batch_model_input(batch, device)
    v_pred = model(x, mask)
    v_true = batch["component_v_kms"].to(device)
    slot_ok = batch["component_valid"].to(device)
    n_slot = slot_ok.sum().clamp(min=1.0)
    slot_v = loss_v(v_pred, v_true)
    total = (slot_v * slot_ok).sum() / n_slot
    parts = {"slot_v": float(total.item()), "total": float(total.item())}
    return total, parts


def train_scheme_d_lite_oracle(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=8,
    log_every=200,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """Scheme D-lite Phase 1: oracle-K velocity-center regression only."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_v = nn.SmoothL1Loss(reduction="none")
    n_steps = len(train_loader)
    print(f"[D-lite] Starting: {epochs} epochs, {n_steps} steps/epoch")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_total = 0.0
        n_loss = 0
        print(f"[D-lite] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            loss, parts = _scheme_d_lite_oracle_batch_loss(
                model, batch, device=device, loss_v=loss_v
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += parts["total"]
            sum_total += parts["total"]
            n_loss += 1
            if step % log_every == 0:
                print(f"  [D-lite]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        metrics = eval_scheme_d_lite_oracle_metrics(model, val_loader, device=device)
        print(f"  [D-lite]   val_mae_v {metrics['mae_v']:.3f}")
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_total / max(1, n_loss)))
            history.setdefault("val_mae_v", []).append(metrics["mae_v"])
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_scheme_d_lite_oracle_metrics(model, val_loader, *, device="cpu") -> dict[str, float]:
    model.eval()
    sum_v = 0.0
    n_slot = 0.0
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        v_pred = model(x, mask)
        v_true = batch["component_v_kms"].to(device)
        slot_ok = batch["component_valid"].to(device)
        n_slot += float(slot_ok.sum().item())
        sum_v += ((v_pred - v_true).abs() * slot_ok).sum().item()
    denom = max(1.0, n_slot)
    return {"mae_v": float(sum_v / denom)}


def _dlite_k_pred_from_out(model, k_out: torch.Tensor) -> torch.Tensor:
    """Integer K_pred from CE logits or regression scalar."""
    Kmax = int(getattr(model, "Kmax", 6))
    if getattr(model, "k_mode", "ce") == "reg":
        return torch.clamp(torch.round(k_out), 0, Kmax).long()
    return k_out.argmax(dim=1)


def _scheme_d_lite_batch_loss(
    model,
    batch,
    *,
    device,
    loss_k: nn.Module,
    loss_v: nn.Module,
    w_v: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, mask = batch_model_input(batch, device)
    k_out, v_pred = model(x, mask)
    v_true = batch["component_v_kms"].to(device)
    slot_ok = batch["component_valid"].to(device)
    n_slot = slot_ok.sum().clamp(min=1.0)
    if getattr(model, "k_mode", "ce") == "reg":
        k_true = batch["K_true"].to(device).float().squeeze(-1)
        loss_k_val = loss_k(k_out, k_true)
    else:
        k_true = batch["K_true"].to(device).long().squeeze(-1)
        loss_k_val = loss_k(k_out, k_true)
    slot_v = loss_v(v_pred, v_true)
    loss_v_val = (slot_v * slot_ok).sum() / n_slot
    total = loss_k_val + float(w_v) * loss_v_val
    parts = {
        "loss_k": float(loss_k_val.item()),
        "loss_v": float(loss_v_val.item()),
        "total": float(total.item()),
    }
    return total, parts


def train_scheme_d_lite(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=12,
    log_every=200,
    w_v=1.0,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """D-lite two-stage: K head (CE or SmoothL1) + velocity slots (non-oracle)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    k_mode = getattr(model, "k_mode", "ce")
    loss_k = (
        nn.SmoothL1Loss(beta=1.0)
        if k_mode == "reg"
        else nn.CrossEntropyLoss()
    )
    loss_v = nn.SmoothL1Loss(reduction="none")
    n_steps = len(train_loader)
    print(
        f"[D-lite-2stg] Starting: {epochs} epochs, {n_steps} steps/epoch, "
        f"w_v={w_v}, k_mode={k_mode}"
    )
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_total = 0.0
        n_loss = 0
        print(f"[D-lite-2stg] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            loss, parts = _scheme_d_lite_batch_loss(
                model, batch, device=device, loss_k=loss_k, loss_v=loss_v, w_v=w_v
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += parts["total"]
            sum_total += parts["total"]
            n_loss += 1
            if step % log_every == 0:
                print(f"  [D-lite-2stg]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        metrics = eval_scheme_d_lite_metrics(model, val_loader, device=device)
        print(
            f"  [D-lite-2stg]   val_K_MAE {metrics['mae_k']:.3f}  "
            f"mae_v_oracle {metrics['mae_v_oracle']:.3f}  "
            f"mae_v_predK {metrics['mae_v_pred_k']:.3f}"
        )
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_total / max(1, n_loss)))
            history.setdefault("val_mae_k", []).append(metrics["mae_k"])
            history.setdefault("val_mae_v_oracle", []).append(metrics["mae_v_oracle"])
            history.setdefault("val_mae_v_pred_k", []).append(metrics["mae_v_pred_k"])
            history.setdefault("val_k_acc", []).append(metrics["k_acc"])
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_scheme_d_lite_metrics(model, val_loader, *, device="cpu") -> dict[str, float]:
    """
    Non-oracle D-lite metrics:
      mae_k       - |K_pred - K_true| (argmax for CE, round for reg)
      mae_v_oracle - |dv| on true active slots (comparable to oracle D-lite)
      mae_v_pred_k - |dv| on slots j < min(K_true, K_pred) with true slot valid
      k_acc       - fraction exact K match
    """
    model.eval()
    sum_k = sum_v_oracle = sum_v_predk = 0.0
    n_slot_oracle = n_slot_predk = 0.0
    n_spec = 0
    n_k_exact = 0
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        k_out, v_pred = model(x, mask)
        k_true = batch["K_true"].to(device).long().squeeze(-1)
        k_pred = _dlite_k_pred_from_out(model, k_out)
        v_true = batch["component_v_kms"].to(device)
        slot_ok = batch["component_valid"].to(device)
        bs = x.size(0)
        n_spec += bs
        n_k_exact += int((k_pred == k_true).sum().item())
        sum_k += (k_pred - k_true).abs().sum().item()
        n_slot_oracle += float(slot_ok.sum().item())
        sum_v_oracle += ((v_pred - v_true).abs() * slot_ok).sum().item()
        for i in range(bs):
            kt = int(k_true[i].item())
            kp = int(k_pred[i].item())
            n_use = min(kt, kp)
            if n_use <= 0:
                continue
            for j in range(n_use):
                if slot_ok[i, j] < 0.5:
                    continue
                n_slot_predk += 1.0
                sum_v_predk += float((v_pred[i, j] - v_true[i, j]).abs().item())
    return {
        "mae_k": float(sum_k / max(1, n_spec)),
        "k_acc": float(n_k_exact / max(1, n_spec)),
        "mae_v_oracle": float(sum_v_oracle / max(1.0, n_slot_oracle)),
        "mae_v_pred_k": float(sum_v_predk / max(1.0, n_slot_predk)),
    }


### -------------------- Center heatmap (per-channel P(center)) --------------------


def build_center_target_map(
    v_centers: torch.Tensor,
    valid: torch.Tensor,
    v_axis: torch.Tensor,
    *,
    label_sigma_kms: float,
) -> torch.Tensor:
    """
    Soft center heatmap target: Gaussian splat (peak 1.0) at each valid component center.

    v_centers: (B, Kmax) km/s. valid: (B, Kmax) 1=active slot. v_axis: (C,) km/s.
    Returns (B, C) in [0, 1]. The channel nearest each center is set to exactly 1.0 (CenterNet
    convention), with a Gaussian falloff around it; overlapping components take the elementwise max.
    """
    B, Kmax = v_centers.shape
    C = int(v_axis.shape[0])
    sig = float(label_sigma_kms)
    ### (B, Kmax, C) distance from each center to every channel.
    diff = v_axis.view(1, 1, C) - v_centers.view(B, Kmax, 1)
    g = torch.exp(-0.5 * (diff / max(sig, 1e-6)) ** 2)
    g = g * valid.view(B, Kmax, 1)
    target = g.max(dim=1).values
    ### Force exact peak at the nearest channel to each active center.
    nearest = diff.abs().argmin(dim=2)  # (B, Kmax)
    b_idx = torch.arange(B, device=v_centers.device).view(B, 1).expand(B, Kmax)
    active = valid > 0.5
    if active.any():
        target[b_idx[active], nearest[active]] = 1.0
    return target


def gaussian_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalty-reduced focal loss on a Gaussian heatmap target (CornerNet/CenterNet).

    Positives are channels with target == 1 (the center peaks); everywhere else is a negative
    whose penalty is down-weighted by (1 - target)^beta so near-peak channels are not punished
    hard. Normalized by the number of positive peaks.
    """
    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos = (target >= 1.0).float()
    neg = 1.0 - pos
    if valid_mask is not None:
        pos = pos * valid_mask
        neg = neg * valid_mask
    pos_loss = -torch.log(pred) * (1.0 - pred) ** alpha * pos
    neg_loss = -torch.log(1.0 - pred) * pred ** alpha * (1.0 - target) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney). Returns 0.5 if a class is absent."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ### Tie-averaged ranks (Mann-Whitney).
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg = np.empty(counts.size)
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = avg[inv]
    sum_pos = ranks[labels].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _center_heatmap_batch(model, batch, v_axis, device, label_sigma_kms):
    x, mask = batch_model_input(batch, device)
    logits = model(x, mask)
    target = build_center_target_map(
        batch["component_v_kms"].to(device),
        batch["component_valid"].to(device),
        v_axis,
        label_sigma_kms=label_sigma_kms,
    )
    loss = gaussian_focal_loss(logits, target, mask)
    return logits, target, loss


def train_center_heatmap(
    model,
    train_loader,
    val_loader,
    v_axis,
    *,
    device="cpu",
    lr=1e-3,
    epochs=12,
    log_every=200,
    label_sigma_kms=4.0,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """Train the per-channel center heatmap (no peak decoder; decoder-free proxy metrics)."""
    model.to(device)
    v_axis = torch.as_tensor(np.asarray(v_axis), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    n_steps = len(train_loader)
    print(f"[heatmap] Starting: {epochs} epochs, {n_steps} steps/epoch, label_sigma={label_sigma_kms} km/s")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        sum_total = 0.0
        n_loss = 0
        print(f"[heatmap] Epoch {ep}/{epochs}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            _, _, loss = _center_heatmap_batch(model, batch, v_axis, device, label_sigma_kms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.item())
            sum_total += float(loss.item())
            n_loss += 1
            if step % log_every == 0:
                print(f"  [heatmap]   step {step}/{n_steps}  train_loss {running/log_every:.4f}", flush=True)
                running = 0.0
        metrics = eval_center_heatmap_metrics(model, val_loader, v_axis, device=device, label_sigma_kms=label_sigma_kms)
        print(
            f"  [heatmap]   val_loss {metrics['loss']:.4f}  "
            f"pos_prob {metrics['pos_prob_mean']:.3f}  neg_prob {metrics['neg_prob_mean']:.3f}  "
            f"auc {metrics['channel_auc']:.3f}",
            flush=True,
        )
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_total / max(1, n_loss)))
            history.setdefault("val_loss", []).append(metrics["loss"])
            history.setdefault("val_pos_prob", []).append(metrics["pos_prob_mean"])
            history.setdefault("val_neg_prob", []).append(metrics["neg_prob_mean"])
            history.setdefault("val_auc", []).append(metrics["channel_auc"])
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_center_heatmap_metrics(
    model, val_loader, v_axis, *, device="cpu", label_sigma_kms=4.0
) -> dict[str, float]:
    """
    Decoder-free heatmap metrics (no peak selection):
      loss          - mean gaussian focal loss
      pos_prob_mean - mean sigmoid at true center peaks (target == 1)
      neg_prob_mean - mean sigmoid at valid off-center channels (target < 0.5)
      channel_auc   - rank AUC separating near-center (target >= 0.5) from background
    """
    model.eval()
    v_axis = torch.as_tensor(np.asarray(v_axis), dtype=torch.float32, device=device)
    sum_loss = 0.0
    n_batch = 0
    pos_sum = neg_sum = 0.0
    n_pos = n_neg = 0
    auc_scores: list[np.ndarray] = []
    auc_labels: list[np.ndarray] = []
    for batch in val_loader:
        logits, target, loss = _center_heatmap_batch(model, batch, v_axis, device, label_sigma_kms)
        _, mask = batch_model_input(batch, device)
        prob = torch.sigmoid(logits)
        valid = mask > 0.5 if mask is not None else torch.ones_like(prob, dtype=torch.bool)
        sum_loss += float(loss.item())
        n_batch += 1
        peak = (target >= 1.0) & valid
        offc = (target < 0.5) & valid
        pos_sum += float(prob[peak].sum().item())
        neg_sum += float(prob[offc].sum().item())
        n_pos += int(peak.sum().item())
        n_neg += int(offc.sum().item())
        near = (target >= 0.5) & valid
        far = (target < 0.5) & valid
        sel = near | far
        auc_scores.append(prob[sel].detach().cpu().numpy())
        auc_labels.append(near[sel].detach().cpu().numpy())
    scores = np.concatenate(auc_scores) if auc_scores else np.zeros(0)
    labels = np.concatenate(auc_labels) if auc_labels else np.zeros(0, dtype=bool)
    ### Cap AUC sample for speed on large val sets.
    if scores.size > 200_000:
        idx = np.random.default_rng(0).choice(scores.size, 200_000, replace=False)
        scores, labels = scores[idx], labels[idx]
    return {
        "loss": float(sum_loss / max(1, n_batch)),
        "pos_prob_mean": float(pos_sum / max(1, n_pos)),
        "neg_prob_mean": float(neg_sum / max(1, n_neg)),
        "channel_auc": _binary_auc(scores, labels),
    }


def train_heatmap_count(
    model,
    train_loader,
    val_loader,
    *,
    device="cpu",
    lr=1e-3,
    epochs=8,
    log_every=200,
    Kmax=10,
    grad_clip=None,
    use_scheduler=False,
    history: dict | None = None,
):
    """
    Stage-2 K head on a heatmap: SmoothL1 on scalar K (Scheme B objective).

    Optimizes only parameters with requires_grad=True (heatmap can be frozen).
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("train_heatmap_count: no trainable parameters (heatmap frozen and no K head?)")
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    n_steps = len(train_loader)
    freeze = bool(getattr(model, "freeze_heatmap", False))
    k_input = str(getattr(model, "k_input", "?"))
    print(
        f"[heatmap_k] Starting: {epochs} epochs, {n_steps} steps/epoch, "
        f"k_input={k_input}, freeze_heatmap={freeze}",
        flush=True,
    )
    for ep in range(1, epochs + 1):
        model.train()
        ### Keep Stage-1 in eval when frozen (BatchNorm stats stay put).
        if freeze and hasattr(model, "heatmap"):
            model.heatmap.eval()
        running = 0.0
        sum_loss = 0.0
        n_loss = 0
        print(f"[heatmap_k] Epoch {ep}/{epochs}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            x, mask = batch_model_input(batch, device)
            K = batch["K_true"].to(device).float().squeeze(-1)
            K_hat = model(x, mask)
            loss = loss_fn(K_hat, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            li = float(loss.item())
            running += li
            sum_loss += li
            n_loss += 1
            if step % log_every == 0:
                print(
                    f"  [heatmap_k]   step {step}/{n_steps}  train_loss {running / log_every:.4f}",
                    flush=True,
                )
                running = 0.0
        metrics = eval_heatmap_count_k(model, val_loader, device=device, Kmax=Kmax)
        print(f"  [heatmap_k]   val_K_MAE {metrics['k_mae']:.3f}", flush=True)
        if history is not None:
            history.setdefault("epoch", []).append(int(ep))
            history.setdefault("train_loss_epoch", []).append(float(sum_loss / max(1, n_loss)))
            history.setdefault("val_K_MAE", []).append(float(metrics["k_mae"]))
            history.setdefault("lr", []).append(float(opt.param_groups[0]["lr"]))
        if sched is not None:
            sched.step()
    return model


@torch.no_grad()
def eval_heatmap_count_k(model, val_loader, *, device="cpu", Kmax=10) -> dict:
    """Round-clamp K head predictions; MAE / exact vs K_true."""
    model.eval()
    abs_err = 0.0
    exact = 0
    n = 0
    for batch in val_loader:
        x, mask = batch_model_input(batch, device)
        K = batch["K_true"].to(device).long().squeeze(-1)
        K_hat = model(x, mask)
        K_pred = torch.clamp(torch.round(K_hat), 0, int(Kmax)).long()
        abs_err += float((K_pred - K).abs().sum().item())
        exact += int((K_pred == K).sum().item())
        n += int(x.size(0))
    return {
        "k_mae": float(abs_err / max(1, n)),
        "k_exact_frac": float(exact / max(1, n)),
        "n": int(n),
    }


### Aliases for Scheme B/C notebooks
train_count = train_scheme_b
train_count_classify = train_scheme_c
