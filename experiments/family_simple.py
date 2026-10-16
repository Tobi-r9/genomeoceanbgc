#!/usr/bin/env python
"""A deliberately minimal method: parts + order + four chemistry numbers.

    sbatch experiments/family_simple.sbatch

Motivated by what the other five methods showed:

  * indicator features (one switch per module) survive a family split, because a
    module never seen in training has a switch that was always off, so the model
    simply has no opinion about it and falls back on the rest of the recipe;
  * sequence features must extrapolate onto unseen residues and can fail badly;
  * of the thesis method's numeric features, only the architecture-compatibility
    counts carry signal. Its `upstream_len`, `downstream_len` and `GC_count` are
    the SAME quantity (number of modules - 1) computed three times, and correlate
    with the outcome at r = 0.001; `GC_mean` is r = 0.009.

So this method is `modules+bigram` plus exactly four numbers:

    upstream_sum, downstream_sum   how many junctions are architecturally
                                   compatible, in each direction
    GC_median, GC_max              GC content summary, capped at 65 %

No sequence, no k-mers, no pretrained model. The four numbers are
**family-agnostic**: "how many junctions fit" and "what is the median GC" mean
the same thing for a familiar recipe and a completely novel one, so they should
transfer where k-mer features do not.

Same protocol as family_eval.py: sweep 19 classifier configurations on VAL, keep
both a per-fold tuned arm and a frozen arm, refit on TRAIN+VAL, score TEST once,
bootstrap the CI.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import (RESULT_DIR, bootstrap_ap, feature_blocks,  # noqa: E402
                               load, model_grid, proba)
from family_thesis import donor_maps, static_features  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(RESULT_DIR)
OUT_DIR = os.path.join(RESULTS_ROOT, "family")
FROZEN = "hgb_lr0.1_l63"          # what the parts-list baselines use
KEEP = [0, 1, 5, 6]               # upstream_sum, downstream_sum, GC_median, GC_max
N_BOOT = 2000


def run_fold(df, emb, fs, family, num):
    t0 = time.time()
    fold = fs[fs["family"] == family]
    d = df.copy()
    d["split"] = d["id"].map(dict(zip(fold["id"], fold["split"])))
    assert d["split"].notna().all(), f"{family}: fold does not cover every id"
    X = np.hstack([feature_blocks(d, emb)["modules+bigram"], num[:, KEEP]])

    tr = (d["split"] == "train").to_numpy()
    va = (d["split"] == "val").to_numpy()
    te = (d["split"] == "test").to_numpy()
    y = d["produced"].to_numpy().astype(int)
    yte = y[te]
    print(f"\n=== {family} simple: train {tr.sum()} val {va.sum()} test {te.sum()} "
          f"| {X.shape[1]} features", flush=True)

    sweep = []
    for mname, factory in model_grid():
        try:
            clf = factory(); clf.fit(X[tr], y[tr])
            sweep.append({"model": mname,
                          "val_ap": float(average_precision_score(y[va], proba(clf, X[va])))})
        except Exception as e:
            print(f"  {mname:20s} FAILED {type(e).__name__}: {e}", flush=True)
    sw = pd.DataFrame(sweep)
    sw.insert(0, "family", family)
    sw.to_csv(os.path.join(OUT_DIR, f"{family}_simple_val_sweep.csv"), index=False)

    grid = dict(model_grid())
    rows, preds = [], {}
    for arm in ("tuned", "frozen"):
        mname = (sw.loc[sw["val_ap"].idxmax(), "model"] if arm == "tuned" else FROZEN)
        clf = grid[mname](); clf.fit(X[tr | va], y[tr | va])
        p = proba(clf, X[te])
        lo, hi = bootstrap_ap(yte, p, n_boot=N_BOOT)
        preds[arm] = p
        rows.append({"family": family, "arm": arm, "method": "simple", "model": mname,
                     "test_ap": float(average_precision_score(yte, p)),
                     "ap_lo": lo, "ap_hi": hi,
                     "test_roc_auc": float(roc_auc_score(yte, p)),
                     "chance_ap": float(yte.mean()), "n_test": int(te.sum()),
                     "test_positives": int(yte.sum())})
        print(f"  [{arm:6s}] {mname:20s} test AP {rows[-1]['test_ap']:.4f} "
              f"[{lo:.4f}, {hi:.4f}] ROC-AUC {rows[-1]['test_roc_auc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{family}_simple.csv"), index=False)
    np.savez(os.path.join(OUT_DIR, f"{family}_simple.npz"),
             ids=d.loc[te, "id"].to_numpy(), y=yte, **preds)
    print(f"  {family} done in {(time.time()-t0)/60:.1f} min", flush=True)


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
                if not os.path.exists(os.path.join(OUT_DIR, f"{f}_simple.csv"))]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} folds -> {todo}", flush=True)
    if not todo:
        return 0

    df, emb = load()
    org, gc, arch = donor_maps()
    mod_lists = [[t.strip() for t in str(s).split(",")] for s in df["modules"]]
    _, num = static_features(mod_lists, org, gc, arch)

    failed = []
    for family in todo:
        try:
            run_fold(df, emb, fs, family, num)
        except Exception as e:
            print(f"!!! {family} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(family)
    if failed:
        print(f"\n!!! {len(failed)} fold(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
