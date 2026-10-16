#!/usr/bin/env python
"""Re-score every method on a DEDUPLICATED test set.

    python experiments/family_dedup.py

Each family fold's test set still contains recipes that have a near-identical
relative in training -- same modules in the same positions except one. Those are
the recipes a model can get right by memorising rather than by understanding, and
their share ("twin rate") varies from 0 % to 96 % across folds.

This script drops them. For every fold it keeps only test constructs with **no**
one-module-substitution neighbour anywhere in that fold's training set, and
recomputes AP and ROC-AUC on what remains.

Nothing is refit. Every method's per-construct test scores were saved when it
ran, so this is a pure post-hoc filter of the evaluation set. The filter is
label-blind: whether a construct has a twin depends only on which modules it
uses, never on whether it produced.

Reading the output: scores here are LOWER than on the full test set for almost
every method, because the recipes that survive are the genuinely novel ones. The
positive rate also shifts, so each fold's chance line moves too and is reported.

Writes results/family/dedup_test.csv
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import bootstrap_ap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
FAM = os.path.join(REPO, "experiments", "results", "family")
MIN_POS = 30          # below this a fold's clean subset is too small to quote


def clean_mask(mod_lists, train_idx, test_idx):
    seen = set()
    for i in train_idx:
        l = mod_lists[i]
        for q in range(len(l)):
            seen.add((tuple(l[:q]), tuple(l[q + 1:])))
    keep = []
    for i in test_idx:
        l = mod_lists[i]
        keep.append(not any((tuple(l[:q]), tuple(l[q + 1:])) in seen for q in range(len(l))))
    return np.array(keep)


def main() -> int:
    d = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    mod_lists = [[t.strip() for t in str(s).split(",")] for s in d["modules"]]
    y_all = d["produced"].to_numpy().astype(int)
    fs = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits.csv"))
    summ = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))

    rows = []
    for _, meta in summ[summ["primary"]].iterrows():
        f = meta["family"]
        fold = fs[fs["family"] == f]
        sp = d["id"].map(dict(zip(fold["id"], fold["split"])))
        tr = np.where((sp == "train").to_numpy())[0]
        te = np.where((sp == "test").to_numpy())[0]
        keep = clean_mask(mod_lists, tr, te)
        ids_te = d["id"].to_numpy()[te]
        yte = y_all[te]
        n_clean, pos_clean = int(keep.sum()), int(yte[keep].sum())
        print(f"{f:12s} test {len(te):5d} -> clean {n_clean:5d} "
              f"({pos_clean} producers, chance {yte[keep].mean():.3f})", flush=True)

        def add(method, arm, ids, scores, yy):
            order = pd.Series(range(len(ids)), index=ids).reindex(ids_te).to_numpy()
            s = scores[order][keep]
            yk = yy[order][keep]
            assert (yk == yte[keep]).all(), f"{f}/{method}: labels disagree after alignment"
            lo, hi = bootstrap_ap(yk, s, n_boot=2000)
            rows.append(dict(family=f, method=method, arm=arm,
                             test_ap=float(average_precision_score(yk, s)),
                             ap_lo=lo, ap_hi=hi,
                             test_roc_auc=float(roc_auc_score(yk, s)),
                             chance_ap=float(yk.mean()), n_test=n_clean,
                             test_positives=pos_clean, leak=float(meta["one_swap_leak_pct"]),
                             enough=bool(pos_clean >= MIN_POS)))

        z = np.load(os.path.join(FAM, f"{f}_scores.npz"))
        for k in z.files:
            if "__" in k:
                arm, meth = k.split("__")
                add(meth, arm, z["ids"], z[k], z["y"])
        for meth, fn in (("symbolic", f"{f}_symbolic.npz"), ("thesis", f"{f}_thesis.npz")):
            p = os.path.join(FAM, fn)
            if os.path.exists(p):
                zz = np.load(p)
                add(meth, "frozen", zz["test_ids"], zz["test_scores"], zz["test_y"])
        p = os.path.join(FAM, f"{f}_simple.npz")
        if os.path.exists(p):
            zz = np.load(p)
            for arm in ("tuned", "frozen"):
                if arm in zz.files:
                    add("simple", arm, zz["ids"], zz[arm], zz["y"])

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(FAM, "dedup_test.csv"), index=False)
    print(f"\nwrote {os.path.join(FAM, 'dedup_test.csv')}  ({len(out)} rows)")
    usable = out[out["enough"]]["family"].nunique()
    print(f"folds whose clean subset keeps >= {MIN_POS} producers: {usable}/12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
