#!/usr/bin/env python
"""Ensemble of the Thesis method and modules+bigram, mixed on validation.

    sbatch experiments/family_ensemble.sbatch

WHY THIS IS WORTH TRYING: the two methods do not see the same information, which
is easy to assume but false.

    modules+bigram  exact module identity (1,173 switches) + which module follows
                    which. No chemistry, no sequence.
    thesis          organism identity (100 switches -- a ~12x COARSENING of module
                    identity, since 1,011 of 1,173 modules share their organism
                    with at least five others), architecture-compatibility counts,
                    GC statistics, and TF-IDF k-mers. It has NO per-module column
                    and NO adjacency feature at all.

So neither is a superset of the other, and they win on different folds. That is
the setup an ensemble is for.

PROTOCOL. Both base models use their default settings, because `thesis` has no
tuned arm; the mixing weight is the only thing chosen, and it is chosen on the
fold's own family-held-out VALIDATION set, then applied unchanged to TEST.

  1. fit each base on TRAIN, score VAL;
  2. search the combination rule and weight that maximise VAL average precision;
  3. fit each base on TRAIN+VAL, score TEST, apply the frozen rule+weight once.

Two combination rules are searched, since the two models' scores are not
necessarily on comparable scales:
  * `prob` -- weighted average of the predicted probabilities;
  * `rank` -- weighted average after converting each model's scores to normalised
    ranks, which is scale-free and usually the safer choice.
w = 0 is pure modules+bigram and w = 1 is pure thesis, so both are inside the
search space and the ensemble can decline to mix.

Outputs results/family/<fold>_ensemble.json and <fold>_ensemble.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import (RESULT_DIR, bootstrap_ap, feature_blocks,  # noqa: E402
                               load, model_grid, proba)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(RESULT_DIR)
OUT_DIR = os.path.join(RESULTS_ROOT, "family")
BASE_MODEL = "hgb_lr0.1_l63"      # modules+bigram default, as used everywhere else
WEIGHTS = np.round(np.arange(0.0, 1.001, 0.05), 3)
N_BOOT = 2000


def norm_rank(x):
    return rankdata(x) / len(x)


def mix(a, b, w, rule):
    """w = 0 -> pure a (modules+bigram); w = 1 -> pure b (thesis)."""
    if rule == "rank":
        a, b = norm_rank(a), norm_rank(b)
    return (1.0 - w) * a + w * b


def run_fold(df, emb, fs, family):
    t0 = time.time()
    th = np.load(os.path.join(OUT_DIR, f"{family}_thesis.npz"))
    fold = fs[fs["family"] == family]
    d = df.copy()
    d["split"] = d["id"].map(dict(zip(fold["id"], fold["split"])))
    assert d["split"].notna().all(), f"{family}: fold does not cover every id"
    X = feature_blocks(d, emb)["modules+bigram"]
    tr = (d["split"] == "train").to_numpy()
    va = (d["split"] == "val").to_numpy()
    te = (d["split"] == "test").to_numpy()
    y = d["produced"].to_numpy().astype(int)
    ids = d["id"].to_numpy()

    grid = dict(model_grid())
    clf = grid[BASE_MODEL](); clf.fit(X[tr], y[tr])
    a_val = proba(clf, X[va])
    clf2 = grid[BASE_MODEL](); clf2.fit(X[tr | va], y[tr | va])
    a_te = proba(clf2, X[te])

    # align thesis scores to our row order
    b_val = pd.Series(th["val_scores"], index=th["val_ids"]).loc[ids[va]].to_numpy()
    b_te = pd.Series(th["test_scores"], index=th["test_ids"]).loc[ids[te]].to_numpy()
    assert (pd.Series(th["val_y"], index=th["val_ids"]).loc[ids[va]].to_numpy() == y[va]).all()
    assert (pd.Series(th["test_y"], index=th["test_ids"]).loc[ids[te]].to_numpy() == y[te]).all()

    # ---- choose rule and weight on VALIDATION only ----------------------
    best = None
    for rule in ("prob", "rank"):
        for w in WEIGHTS:
            ap = average_precision_score(y[va], mix(a_val, b_val, w, rule))
            if best is None or ap > best[0]:
                best = (ap, rule, float(w))
    val_ap, rule, w = best
    print(f"\n=== {family}: chose rule={rule} w={w:.2f} (val AP {val_ap:.4f}); "
          f"pure mod+bigram val AP {average_precision_score(y[va], a_val):.4f}, "
          f"pure thesis val AP {average_precision_score(y[va], b_val):.4f}", flush=True)

    # ---- apply once to TEST ---------------------------------------------
    s_te = mix(a_te, b_te, w, rule)
    lo, hi = bootstrap_ap(y[te], s_te, n_boot=N_BOOT)
    out = {"family": family, "rule": rule, "weight": w, "val_ap": float(val_ap),
           "test": {"ap": float(average_precision_score(y[te], s_te)),
                    "ap_lo": lo, "ap_hi": hi,
                    "roc_auc": float(roc_auc_score(y[te], s_te)),
                    "chance_ap": float(y[te].mean())},
           "base_modules_bigram": {
               "test_ap": float(average_precision_score(y[te], a_te)),
               "test_roc_auc": float(roc_auc_score(y[te], a_te))},
           "base_thesis": {
               "test_ap": float(average_precision_score(y[te], b_te)),
               "test_roc_auc": float(roc_auc_score(y[te], b_te))}}
    print(f"  TEST ensemble AP {out['test']['ap']:.4f} [{lo:.4f}, {hi:.4f}] "
          f"| mod+bigram {out['base_modules_bigram']['test_ap']:.4f} "
          f"| thesis {out['base_thesis']['test_ap']:.4f} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    with open(os.path.join(OUT_DIR, f"{family}_ensemble.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    # VAL scores are saved too, so a single GLOBAL mixing weight can be chosen
    # across folds on validation. Per-fold weight selection turned out to overfit
    # the small validation sets -- see FAMILY_GUIDE.md.
    np.savez(os.path.join(OUT_DIR, f"{family}_ensemble.npz"),
             ids=ids[te], y=y[te], ensemble=s_te, modules_bigram=a_te, thesis=b_te,
             val_ids=ids[va], val_y=y[va], val_modules_bigram=a_val, val_thesis=b_val)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--split-name", default="family",
                    help="which split family to run: reads splits/<name>_splits.csv and writes results/<name>/. Default `family` reproduces the original run.")
    args = ap.parse_args()
    global OUT_DIR
    OUT_DIR = os.path.join(RESULTS_ROOT, args.split_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    fs = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits.csv"))
    summ = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits_summary.csv"))
    fams = args.families or list(summ.sort_values("n_test", ascending=False)["family"])
    todo = fams if args.families else fams[args.shard::args.num_shards]
    if not args.redo:
        todo = [f for f in todo
                if not os.path.exists(os.path.join(OUT_DIR, f"{f}_ensemble.json"))]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} folds -> {todo}", flush=True)
    if not todo:
        return 0

    df, emb = load()
    failed = []
    for family in todo:
        try:
            run_fold(df, emb, fs, family)
        except Exception as e:
            print(f"!!! {family} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(family)
    if failed:
        print(f"\n!!! {len(failed)} fold(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
