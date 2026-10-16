#!/usr/bin/env python
"""A third, stricter evaluation tier: cap similarity to training by Jaccard.

    python experiments/family_jaccard.py [--split-name family]

WHY THIS EXISTS
---------------
The deduplicated tier (family_dedup.py) removes test constructs that have a
training construct differing by exactly one module **at the same length, in the
same positions**. That filter is positional, and it has a blind spot: a training
construct differing by an insertion or deletion slips straight through.

XfaS is the case that exposed it. Its one-swap twin rate is 0.0 %, which made it
look like the strictest fold in the study -- but its test set is almost entirely
6-module constructs while training is dominated by 4-7 module ones, so the
typical nearest neighbour looks like

    test  (6 modules): 1108,112,103,876,877,1245
    train (7 modules): 1108,112,103,876,877,878,879

Five identical modules in order, different length: invisible to the positional
filter. Measured by Jaccard, XfaS is the *least* clean primary fold (median 0.62
against its nearest training construct), not the cleanest.

WHAT THIS DOES
--------------
For every test construct, compute the maximum Jaccard similarity of its module
SET against every training construct in that fold, then keep only those at or
below a threshold. Being set-based, this subsumes one-module substitutions,
insertions, deletions and reorderings in a single criterion.

Three tiers are produced (0.7, 0.6, 0.5) so the degradation is visible rather
than asserted at one arbitrary cut.

Nothing is refit. Every method's per-construct test scores were saved when it
ran, so this is a filter of the evaluation set plus a re-score. The filter is
label-blind: Jaccard depends only on which modules a construct uses.

Writes results/<split>/jaccard_test.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import bootstrap_ap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(REPO, "experiments", "results")
THRESHOLDS = (0.7, 0.6, 0.5)
MIN_POS = 30
ENSEMBLE_W = 0.65        # global weight chosen on validation; see FAMILY_GUIDE.md 8i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", default="family")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    FAM = os.path.join(RESULTS_ROOT, args.split_name)

    d = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    mod_sets = [sorted({t.strip() for t in str(s).split(",")}) for s in d["modules"]]
    ids_all = d["id"].to_numpy()
    vocab = sorted({m for l in mod_sets for m in l})
    vi = {m: j for j, m in enumerate(vocab)}
    A = sp.csr_matrix(
        (np.ones(sum(map(len, mod_sets)), np.int32),
         ([i for i, l in enumerate(mod_sets) for _ in l], [vi[m] for l in mod_sets for m in l])),
        shape=(len(mod_sets), len(vocab)))
    sizes = np.asarray(A.sum(1)).ravel()

    fs = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits.csv"))
    summ = pd.read_csv(os.path.join(SPLIT_DIR, f"{args.split_name}_splits_summary.csv"))

    rows = []
    for _, meta in summ[summ["primary"]].iterrows():
        f = meta["family"]
        fold = fs[fs["family"] == f]
        sp_ = d["id"].map(dict(zip(fold["id"], fold["split"])))
        tr = np.where((sp_ == "train").to_numpy())[0]
        te = np.where((sp_ == "test").to_numpy())[0]
        # max Jaccard of each test construct against ANY training construct
        At, st = A[tr], sizes[tr]
        best = np.zeros(len(te))
        for s0 in range(0, len(te), 500):
            r = te[s0:s0 + 500]
            B = (A[r] @ At.T).tocoo()
            if B.nnz:
                jac = B.data / (sizes[r][B.row] + st[B.col] - B.data)
                np.maximum.at(best[s0:s0 + 500], B.row, jac)
        ids_te = ids_all[te]

        scores = {}
        p = os.path.join(FAM, f"{f}_scores.npz")
        if os.path.exists(p):
            z = np.load(p)
            order = pd.Series(range(len(z["ids"])), index=z["ids"]).reindex(ids_te).to_numpy()
            for k in z.files:
                if "__" in k:
                    arm, meth = k.split("__")
                    scores[(meth, arm)] = (z[k][order], z["y"][order])
        for meth, fn in (("symbolic", f"{f}_symbolic.npz"), ("thesis", f"{f}_thesis.npz")):
            q = os.path.join(FAM, fn)
            if os.path.exists(q):
                z = np.load(q)
                o = pd.Series(range(len(z["test_ids"])), index=z["test_ids"]).reindex(ids_te).to_numpy()
                scores[(meth, "frozen")] = (z["test_scores"][o], z["test_y"][o])
        q = os.path.join(FAM, f"{f}_simple.npz")
        if os.path.exists(q):
            z = np.load(q)
            o = pd.Series(range(len(z["ids"])), index=z["ids"]).reindex(ids_te).to_numpy()
            for arm in ("tuned", "frozen"):
                if arm in z.files:
                    scores[("simple", arm)] = (z[arm][o], z["y"][o])
        q = os.path.join(FAM, f"{f}_ensemble.npz")
        if os.path.exists(q):
            z = np.load(q)
            o = pd.Series(range(len(z["ids"])), index=z["ids"]).reindex(ids_te).to_numpy()
            a, b, yy = z["modules_bigram"][o], z["thesis"][o], z["y"][o]
            mix = (1 - ENSEMBLE_W) * (rankdata(a) / len(a)) + ENSEMBLE_W * (rankdata(b) / len(b))
            scores[("ensemble", "frozen")] = (mix, yy)

        for thr in THRESHOLDS:
            keep = best <= thr
            n, npos = int(keep.sum()), int(scores[list(scores)[0]][1][keep].sum())
            print(f"{f:<28s} J<={thr}: {n:5d} constructs, {npos:4d} producers", flush=True)
            if n < 20 or npos < 5:
                continue
            for (meth, arm), (s, yy) in scores.items():
                yk, sk = yy[keep], s[keep]
                if yk.sum() == 0 or yk.sum() == len(yk):
                    continue
                # the ensemble mixes ranks computed on the FULL test set, then filters;
                # that is deliberate -- re-ranking inside the subset would change the model
                lo, hi = bootstrap_ap(yk, sk, n_boot=args.n_boot)
                rows.append(dict(family=f, threshold=thr, method=meth, arm=arm,
                                 test_ap=float(average_precision_score(yk, sk)),
                                 ap_lo=lo, ap_hi=hi,
                                 test_roc_auc=float(roc_auc_score(yk, sk)),
                                 chance_ap=float(yk.mean()), n_test=n, test_positives=npos,
                                 orig_n_test=int(len(te)), orig_leak=float(meta["one_swap_leak_pct"]),
                                 enough=bool(npos >= MIN_POS)))
    out = pd.DataFrame(rows)
    dst = os.path.join(FAM, "jaccard_test.csv")
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst} ({len(out)} rows)")
    for thr in THRESHOLDS:
        sub = out[(out["threshold"] == thr) & out["enough"]]
        print(f"  J<={thr}: {sub['family'].nunique()} folds keep >= {MIN_POS} producers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
