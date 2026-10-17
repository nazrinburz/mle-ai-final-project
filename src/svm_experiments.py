"""
svm_experiments.py - hyperparameter study for the Pegasos SVM (Problem 3b, 4b).

Usage from run_all.py (use the same splits as every other model):

    from .svm_experiments import run_svm_study
    best = run_svm_study(X_tr, y_tr, X_val, y_val).    # y = 0/1 price tier

Everything is chosen on validation data only (no test-set peeking).
Figures are saved to report/figures/. Raw (unscaled) features are fine:
PegasosSVM scales with training statistics internally.

Run directly (python -m src.svm_experiments) for a demo on synthetic data.
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")   # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np

from .svm import PegasosSVM

def _accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))

def _f1(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = np.sum((y == 1) & (p == 1))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

def _evaluate(model, X_tr, y_tr, X_val, y_val):
    p = model.predict(X_val)
    return{
        "val_acc": _accuracy(y_val, p), 
        "val_f1": _f1(y_val, p),
        "final_J": float(model.objective_history[-1, 1]), 
        "norm_w": float(np.linalg.norm(model.w)), 
        "margin": model.margin_width_, 
        "n_sv": int(len(model.support_vector_indices())),
    }

def _print_table(title, rows, key_name):
    print(f"\n{title}")
    print(f"{key_name:>12} | val_acc | val_f1 | final_J | ||w|| | margin | #SV")
    for key, r in rows:
        print(f"{str(key):>12} | {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['final_J']:.4f} | {r['norm_w']:<8.3f} | {r['margin']:<8.3f} | {r['n_sv']}")

def run_svm_study(
        X_tr, y_tr, X_val, y_val,
        lambdas = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
        iters = (1_000, 5_000, 20_000, 50_000), 
        gammas = (0.01, 0.1, 1.0), 
        n_rff=300, 
        base_iters=20_000,
        out_dir="report/figures", 
        seed = 42,
):
    """Running all sweeps, print tables, save figures, return the best configs (by val F1)."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. lambda sweep (linear SVM): margin vs validation metric
    lam_rows = []
    for lam in lambdas:
        m = PegasosSVM(lambda_=lam, n_iters=base_iters, random_state=seed).fit(X_tr, y_tr)
        lam_rows.append((lam, _evaluate(m, X_tr, y_tr, X_val, y_val)))
    _print_table("Lambda sweep (linear, 'pegasos' schedule)", lam_rows, "lambda")
    best_lam = max(lam_rows, key=lambda kv: kv[1]["val_f1"])[0]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.semilogx(lambdas, [r["margin"] for _, r in lam_rows], "o-", color="tab:blue")
    ax1.set_yscale("log")
    ax1.set_xlabel("lambda"); ax1.set_ylabel("margin width 2/||w||", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.semilogx(lambdas, [r["val_acc"] for _, r in lam_rows], "s--", color="tab:red", label="val acc")
    ax2.semilogx(lambdas, [r["val_f1"] for _, r in lam_rows], "^--", color="tab:green", label="val F1")
    ax2.set_ylabel("validation metric"); ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax1.set_title("SVM: margin vs validation metric")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "svm_lambda_sweep.png"), dpi=150); plt.close(fig)

    # 2. number of iterations / epochs
    it_rows = []
    for n_it in iters:
        m = PegasosSVM(lambda_=best_lam, n_iters=n_it, random_state=seed).fit(X_tr, y_tr)
        it_rows.append((n_it, _evaluate(m, X_tr, y_tr, X_val, y_val)))
    _print_table(f"Iterations sweep (lambda={best_lam:g}; epochs = iters / {len(y_tr)})", it_rows, "n_iters")

    # 3. learning-rate schedule: objective curves + margin
    fig, ax = plt.subplots(figsize=(6, 4))
    sched_rows = []
    for sched in ("pegasos", "inv_sqrt"):
        m = PegasosSVM(lambda_=best_lam, n_iters=base_iters, random_state=seed, lr_schedule=sched).fit(X_tr, y_tr)
        sched_rows.append((sched, _evaluate(m, X_tr, y_tr, X_val, y_val)))
        ax.plot(m.objective_history[:, 0], m.objective_history[:, 1], label=sched)
    _print_table(f"Learning-rate schedule (lambda={best_lam:g})", sched_rows, "schedule")
    ax.set_xlabel("iteration"); ax.set_ylabel("objective J(w, b)")
    ax.set_yscale("log"); ax.legend(); ax.set_title("Objective convergence by schedule")
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "svm_objective_convergence.png"), dpi=150); plt.close(fig)

    # 4. Non-linearity: Random Fourier Features, sweep gamma
    g_rows = []
    for g in gammas:
        m = PegasosSVM(lambda_=best_lam, n_iters=base_iters, random_state=seed, n_rff=n_rff, gamma=g).fit(X_tr, y_tr)
        g_rows.append((g, _evaluate(m, X_tr, y_tr, X_val, y_val)))
    _print_table(f"RFF sweep (n_rff={n_rff}, lambda={best_lam:g}); linear val_f1={dict(lam_rows)[best_lam]['val_f1']:.4f}", g_rows, "gamma")
    best_gamma = max(g_rows, key=lambda kv: kv[1]["val_f1"])[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(gammas, [r["val_f1"] for _, r in g_rows], "o-", label="RFF val F1")
    ax.axhline(dict(lam_rows)[best_lam]["val_f1"], color="gray", ls="--", label="linear val F1")
    ax.set_xlabel("gamma (RBF width)"); ax.set_ylabel("validation F1"); ax.legend()
    ax.set_title("Non-linear SVM (random Fourier features)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "svm_rff_gamma_sweep.png"), dpi=150); plt.close(fig)

    best = {"lambda_": best_lam, "gamma": best_gamma, "n_rff": n_rff}
    print(f"\nBest by validation F1 -> lambda={best_lam:g}, RFF gamma={best_gamma:g}")
    print(f"Figures saved to {out_dir}/")
    return best

if __name__ == "__main__":
    # Demo on non-linear data (use real splits via run_all.py)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4)) * [1, 10, 100, 1000]
    z = X / [1, 10, 100, 1000]
    y = ((z[:, 0] ** 2 + z[:, 1] ** 2) > 1.8).astype(int)
    run_svm_study(X[:1400], y[:1400], X[1400:1700], y[1400:1700], out_dir="report/figures")