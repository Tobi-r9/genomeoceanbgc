#!/usr/bin/env python
"""Uncertainty analysis on VALIDATION for the top sweep configurations.

    python experiments/analyze_val.py

The sweep's argmax over 190 configurations implies more precision than 2,398
validation rows with 339 positives can support. This refits a shortlist, puts a
bootstrap CI on each AP, and runs a **paired** bootstrap on the differences that
actually matter -- embeddings versus the cheap baselines -- so model selection and
the report rest on whether a gap is real, not on the ranking alone.

Test set is not touched.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import (RESULT_DIR, SEED, bootstrap_ap, feature_blocks,  # noqa: E402
                               load, model_grid, proba, scores)

# Shortlist: the best config per feature block (from val_sweep_merged.csv).
SHORTLIST = [
    ("emb_mean+modules", "hgb_lr0.05_l31"),
    ("emb_mean+aac", "logreg_C10.0"),
    ("emb_mean+len", "logreg_C10.0"),
    ("emb_mean", "logreg_C10.0"),
    ("emb_mean_l2", "logreg_C10.0"),
    ("emb_last", "mlp_64_a0.0001"),
    ("modules+bigram", "hgb_lr0.1_l63"),
    ("modules", "hgb_lr0.1_l63"),
    ("modules_bigram", "mlp_256x64_a0.01"),
    ("aac", "hgb_lr0.1_l63"),
    ("len", "hgb_lr0.1_l31"),
]
N_BOOT = 2000


def paired_bootstrap(y, pa, pb, n_boot=N_BOOT, seed=SEED):
    """CI and P(a>b) for AP(a) - AP(b) on the same resampled rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], pa[idx])
                     - average_precision_score(y[idx], pb[idx]))
    d = np.array(diffs)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d > 0).mean())


def main() -> int:
    df, emb = load()
    blocks = feature_blocks(df, emb)
    grid = dict(model_grid())
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    y = df["produced"].to_numpy().astype(int)
    yva = y[va]

    preds, rows = {}, []
    for bname, mname in SHORTLIST:
        clf = grid[mname]()
        clf.fit(blocks[bname][tr], y[tr])
        p = proba(clf, blocks[bname][va])
        preds[(bname, mname)] = p
        s = scores(yva, p)
        lo, hi = bootstrap_ap(yva, p, n_boot=N_BOOT)
        rows.append({"block": bname, "model": mname,
                     "n_features": blocks[bname].shape[1],
                     "val_ap": s["ap"], "ap_lo": lo, "ap_hi": hi,
                     "val_roc_auc": s["roc_auc"]})
        print(f"{bname:22s} {mname:20s} AP {s['ap']:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  ROC-AUC {s['roc_auc']:.4f}", flush=True)

    res = pd.DataFrame(rows).sort_values("val_ap", ascending=False)
    out = os.path.join(RESULT_DIR, "val_shortlist_ci.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"\nchance-level AP (positive rate) = {yva.mean():.4f}")

    # ---- the comparisons the conclusion depends on -------------------------
    best = ("emb_mean", "logreg_C10.0")     # simplest strong embedding config
    print(f"\n=== paired bootstrap vs {best[0]}/{best[1]} ===")
    for ref in [("modules+bigram", "hgb_lr0.1_l63"), ("modules", "hgb_lr0.1_l63"),
                ("aac", "hgb_lr0.1_l63"), ("emb_last", "mlp_64_a0.0001"),
                ("len", "hgb_lr0.1_l31")]:
        m, lo, hi, pw = paired_bootstrap(yva, preds[best], preds[ref])
        verdict = "SIGNIFICANT" if lo > 0 else "not significant"
        print(f"  vs {ref[0]:10s} dAP {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"P(better)={pw:.3f}  {verdict}")

    print("\n=== paired bootstrap: top config vs the simplest strong one ===")
    top = ("emb_mean+modules", "hgb_lr0.05_l31")
    m, lo, hi, pw = paired_bootstrap(yva, preds[top], preds[best])
    print(f"  {top[0]} vs {best[0]}: dAP {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"P(better)={pw:.3f}")
    print("  -> if the interval spans 0 the two are indistinguishable on validation,")
    print("     so prefer the simpler/cheaper model for the final test evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
