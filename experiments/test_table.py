#!/usr/bin/env python
"""Final TEST table: the four headline methods, side by side.

    python experiments/test_table.py

Reports test average precision with a 95% bootstrap CI, plus test ROC-AUC, for

    emb_mean         GenomeOcean-4B mean-pooled embedding + logistic regression
    modules+bigram   multi-hot modules and adjacent-module pairs + HGB
    modules          multi-hot modules only + HGB
    symbolic         the contributed hand-engineered protein classifier

Each model uses the configuration that won on VALIDATION (see
`results/val_shortlist_ci.csv`); nothing is tuned here. Every model is refit on
TRAIN+VAL and scored once on TEST, which is the protocol already used for
`emb_mean` and `symbolic`. `symbolic` scores are reused from
`results/symbolic_scores.npz` rather than refit, and are aligned to the test
frame by row id.

The CI is a percentile bootstrap over test rows: resample the 2,397 test rows
with replacement 2,000 times, recompute AP each time, take the 2.5th and 97.5th
percentiles. It answers "how much would this number move on a different sample
of constructs of the same size", which is the only honest way to read a
0.01 gap between two methods.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import (RESULT_DIR, SEED, bootstrap_ap, feature_blocks,  # noqa: E402
                               load, model_grid, proba)
from analyze_val import paired_bootstrap  # noqa: E402

N_BOOT = 2000
SYMBOLIC_CSV = os.path.join(
    os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
    "bgc_clf", "symbolic", "test.csv")

# block -> (model key, one-line description for the table)
METHODS = [
    ("emb_mean", "logreg_C10.0",
     "GenomeOcean-4B mean-pooled embedding (3072-d), logistic regression C=10"),
    ("modules+bigram", "hgb_lr0.1_l63",
     "which modules are present + which adjacent pairs, gradient boosting"),
    ("modules", "hgb_lr0.1_l63",
     "which modules are present (order-blind), gradient boosting"),
]


def symbolic_test_scores(df_test):
    """Load the contributed model's test scores, aligned to df_test row order."""
    z = np.load(os.path.join(RESULT_DIR, "symbolic_scores.npz"))
    ids = pd.read_csv(SYMBOLIC_CSV, usecols=["id", "produced"])
    assert len(ids) == len(z["test_scores"]) == len(df_test), "test size mismatch"
    s = pd.Series(z["test_scores"], index=ids["id"].to_numpy())
    aligned = s.loc[df_test["id"].to_numpy()].to_numpy()
    # Labels must agree row-for-row, otherwise the alignment is wrong.
    y_sym = pd.Series(z["test_y"], index=ids["id"].to_numpy()).loc[
        df_test["id"].to_numpy()].to_numpy()
    assert (y_sym == df_test["produced"].to_numpy().astype(int)).all(), \
        "symbolic labels do not match the test frame after id alignment"
    return aligned


def main() -> int:
    df, emb = load()
    blocks = feature_blocks(df, emb)
    grid = dict(model_grid())
    tv = df["split"].isin(["train", "val"]).to_numpy()
    te = (df["split"] == "test").to_numpy()
    y = df["produced"].to_numpy().astype(int)
    yte = y[te]

    preds, rows = {}, []
    for bname, mname, desc in METHODS:
        clf = grid[mname]()
        clf.fit(blocks[bname][tv], y[tv])
        p = proba(clf, blocks[bname][te])
        preds[bname] = p
        lo, hi = bootstrap_ap(yte, p, n_boot=N_BOOT)
        rows.append({"method": bname, "model": mname,
                     "n_features": blocks[bname].shape[1],
                     "test_ap": float(average_precision_score(yte, p)),
                     "ap_lo": lo, "ap_hi": hi,
                     "test_roc_auc": float(roc_auc_score(yte, p)),
                     "description": desc})
        print(f"{bname:16s} AP {rows[-1]['test_ap']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"ROC-AUC {rows[-1]['test_roc_auc']:.4f}", flush=True)

    p = symbolic_test_scores(df[te])
    preds["symbolic"] = p
    lo, hi = bootstrap_ap(yte, p, n_boot=N_BOOT)
    rows.append({"method": "symbolic", "model": "n_kmers150000_l23.0_it2500",
                 "n_features": np.nan,
                 "test_ap": float(average_precision_score(yte, p)),
                 "ap_lo": lo, "ap_hi": hi,
                 "test_roc_auc": float(roc_auc_score(yte, p)),
                 "description": "hand-designed protein features "
                                "(composition + NRPS motifs + hashed 6-mers + "
                                "terminal signatures), logistic regression"})
    print(f"{'symbolic':16s} AP {rows[-1]['test_ap']:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"ROC-AUC {rows[-1]['test_roc_auc']:.4f}")

    res = pd.DataFrame(rows).sort_values("test_ap", ascending=False)
    out = os.path.join(RESULT_DIR, "test_table.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"chance-level AP (test positive rate) = {yte.mean():.4f}")

    # Pairwise gaps against the embedding, on the same resampled rows.
    print("\n=== paired bootstrap vs emb_mean (test) ===")
    for name in ("modules+bigram", "modules", "symbolic"):
        m, lo, hi, pw = paired_bootstrap(yte, preds["emb_mean"], preds[name],
                                         n_boot=N_BOOT)
        verdict = "SIGNIFICANT" if lo > 0 else "not significant"
        print(f"  emb_mean - {name:15s} dAP {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"P(emb better)={pw:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
