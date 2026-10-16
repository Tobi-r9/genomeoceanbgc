#!/usr/bin/env python
"""Gather every method's per-fold test numbers for one split scheme.

    python experiments/family_collect.py --split-name {family,genus,multifamily}

Produces results/<split>/all_methods_ci.csv with one row per
(fold, arm, method): test AP with a 2,000-resample bootstrap CI, test ROC-AUC,
the fold's chance line, and the fold metadata needed to read it.

Methods gathered, wherever their output exists:
    emb_mean, modules+bigram, modules   from <fold>_scores.npz  (tuned + frozen)
    symbolic, thesis                    from <fold>_{symbolic,thesis}.npz (frozen only)
    simple                              from <fold>_simple.npz  (tuned + frozen)
    ensemble                            recomputed from <fold>_ensemble.npz at the
                                        GLOBAL validation-chosen weight (0.65 thesis,
                                        rank-averaged) -- not the per-fold weight,
                                        which overfits the small validation sets
    copytwin                            the memorisation floor: copy the label of the
                                        nearest one-module-swap neighbour in training,
                                        or the training base rate if there is none.
                                        Not a model; it is the score a method must beat.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import bootstrap_ap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(REPO, "experiments", "results")
ENSEMBLE_W = 0.65


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", default="family")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    FAM = os.path.join(RESULTS_ROOT, args.split_name)

    d = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    ml = [[t.strip() for t in str(s).split(",")] for s in d["modules"]]
    y_all = d["produced"].to_numpy().astype(int)
    ids_all = d["id"].to_numpy()
    fs = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits.csv"))
    summ = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits_summary.csv"))

    rows = []
    for _, meta in summ.iterrows():
        f = meta["family"]
        fold = fs[fs["family"] == f]
        sp_ = d["id"].map(dict(zip(fold["id"], fold["split"])))
        tr = np.where((sp_ == "train").to_numpy())[0]
        te = np.where((sp_ == "test").to_numpy())[0]
        ids_te, yte = ids_all[te], y_all[te]

        def align(ids, vals):
            o = pd.Series(range(len(ids)), index=ids).reindex(ids_te).to_numpy()
            return vals[o]

        got = {}
        p = os.path.join(FAM, f"{f}_scores.npz")
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                if "__" in k:
                    arm, meth = k.split("__")
                    got[(meth, arm)] = align(z["ids"], z[k])
        for meth, fn in (("symbolic", f"{f}_symbolic.npz"), ("thesis", f"{f}_thesis.npz")):
            q = os.path.join(FAM, fn)
            if os.path.exists(q):
                z = np.load(q); got[(meth, "frozen")] = align(z["test_ids"], z["test_scores"])
        q = os.path.join(FAM, f"{f}_simple.npz")
        if os.path.exists(q):
            z = np.load(q)
            for arm in ("tuned", "frozen"):
                if arm in z.files:
                    got[("simple", arm)] = align(z["ids"], z[arm])
        q = os.path.join(FAM, f"{f}_ensemble.npz")
        if os.path.exists(q):
            z = np.load(q)
            a, b = align(z["ids"], z["modules_bigram"]), align(z["ids"], z["thesis"])
            got[("ensemble", "frozen")] = ((1 - ENSEMBLE_W) * rankdata(a) / len(a)
                                           + ENSEMBLE_W * rankdata(b) / len(b))
        # copy-twin memorisation floor
        idx = collections.defaultdict(list)
        for i in tr:
            l = ml[i]
            for k in range(len(l)):
                idx[(tuple(l[:k]), tuple(l[k + 1:]))].append(y_all[i])
        pred = []
        for i in te:
            l = ml[i]; v = []
            for k in range(len(l)):
                v += idx.get((tuple(l[:k]), tuple(l[k + 1:])), [])
            pred.append(np.mean(v) if v else np.nan)
        pred = np.array(pred)
        got[("copytwin", "frozen")] = np.where(np.isnan(pred), yte.mean(), pred)

        for (meth, arm), s in got.items():
            lo, hi = bootstrap_ap(yte, s, n_boot=args.n_boot)
            rows.append(dict(family=f, method=meth, arm=arm,
                             test_ap=float(average_precision_score(yte, s)),
                             ap_lo=lo, ap_hi=hi,
                             test_roc_auc=float(roc_auc_score(yte, s)),
                             chance_ap=float(yte.mean()), n_test=len(te),
                             test_positives=int(yte.sum()),
                             leak=float(meta["one_swap_leak_pct"]),
                             primary=bool(meta["primary"])))
        print(f"  {f:<28s} {len(got)} method-arms", flush=True)

    out = pd.DataFrame(rows)
    dst = os.path.join(FAM, "all_methods_ci.csv")
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst} ({len(out)} rows, {out['family'].nunique()} folds, "
          f"{out['method'].nunique()} methods)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
