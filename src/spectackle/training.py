### train_scheme_b, train_scheme_c (Spectrum normalization + training loops)
import torch
import torch.nn as nn


def _norm(x):
    """Per-spectrum normalization: subtract mean, divide by std."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / (x.std(dim=1, keepdim=True) + 1e-6)


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
):
    """Scheme B: regression → scalar K, SmoothL1 loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    n_steps = len(train_loader)
    print(f"[B] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        print(f"[B] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            x = _norm(batch["spec"].to(device))
            K = batch["K_true"].to(device).float().squeeze(-1)
            K_hat = model(x)
            loss = loss_fn(K_hat, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            if step % log_every == 0:
                print(f"  [B]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        model.eval()
        mae, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = _norm(batch["spec"].to(device))
                K = batch["K_true"].to(device).long().squeeze(-1)
                K_hat = model(x)
                K_pred = torch.clamp(torch.round(K_hat), 0, Kmax).long()
                mae += (K_pred - K).abs().sum().item()
                n += x.size(0)
        print(f"  [B]   val_K_MAE {mae/n:.3f}")
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
):
    """Scheme C: classification → logits 0..Kmax, CrossEntropy loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_steps = len(train_loader)
    print(f"[C] Starting: {epochs} epochs, {n_steps} steps/epoch, log every {log_every}")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        print(f"[C] Epoch {ep}/{epochs}")
        for step, batch in enumerate(train_loader, start=1):
            x = _norm(batch["spec"].to(device))
            K = batch["K_true"].to(device).long().squeeze(-1)
            logits = model(x)
            loss = loss_fn(logits, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % log_every == 0:
                print(f"  [C]   step {step}/{n_steps}  train_loss {running/log_every:.4f}")
                running = 0.0
        model.eval()
        n, n_correct, mae_argmax, mae_expected = 0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = _norm(batch["spec"].to(device))
                K = batch["K_true"].to(device).long().squeeze(-1)
                logits = model(x)
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
    return model


### Aliases for Scheme B/C notebooks
train_count = train_scheme_b
train_count_classify = train_scheme_c
