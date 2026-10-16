#!/usr/bin/env python
"""Confusion tables and balanced 50/50 subsets, per leave-one-family-out fold.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/family_confusion.py                 # all primary folds
    python experiments/family_confusion.py --families GxpS --all-folds

The family-split counterpart of confusion_and_balanced.py, same protocol:
  * validation numbers from the model fit on TRAIN only;
  * the decision threshold is the F1-optimal one on VALIDATION, applied
    unchanged to TEST, alongside the natural 0.5;
  * test numbers from the model refit on TRAIN+VAL;
  * balanced subsets keep every positive plus an equal number of seeded random
    negatives, which makes accuracy interpretable but does NOT make the task
    easier -- precision, AP and accuracy all rise purely from prevalence.

Model is `emb_mean` + logreg_C10.0, the configuration committed in REPORT.md.

Two deliberate differences from confusion_and_balanced.py, both to avoid
touching anything the random-split runs depend on:

  1. **Output paths are new.** confusion_and_balanced.py OVERWRITES
     splits/val_equal.csv and splits/test_equal.csv. This script writes to
     splits/family_equal/<fold>_{val,test}_equal.csv and
     results/family/<fold>_confusion.csv instead, so the random-split files
     stay exactly as they are.
  2. **F1-optimal threshold uses `argmax(f1[:-1])`**, as run_symbolic.py does.
     confusion_and_balanced.py uses `argmax(f1)` over the full array, which
     indexes `thresholds` one past its end if the maximum ever lands on the
     final point of the precision-recall curve. That point has recall 0 and
     therefore F1 0, so it cannot win in practice and the two forms agree on
     every fold here -- but the guarded form is used since some folds have as
     few as 30 test positives and degenerate curves are likelier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confusion_and_balanced import balanced_subset, confusion_report, show  # noqa: E402
from train_classifiers import SEED, feature_blocks, load, model_grid, proba  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
EQUAL_DIR = os.path.join(SPLIT_DIR, "family_equal")
FAM_DIR = os.path.join(REPO, "experiments", "results", "family")
BLOCK, MODEL = "emb_mean", "logreg_C10.0"
MIN_POS_FOR_BALANCED = 25


def run_fold(df, emb, fs, family) -> list[dict]:
    fold = fs[fs["family"] == family]
    d = df.copy()
    d["split"] = d["id"].map(dict(zip(fold["id"], fold["split"])))
    assert d["split"].notna().all(), f"{family}: fold does not cover every id"

    X = feature_blocks(d, emb)[BLOCK]
    y = d["produced"].to_numpy().astype(int)
    ids = d["id"].to_numpy()
    tr = (d["split"] == "train").to_numpy()
    va = (d["split"] == "val").to_numpy()
    te = (d["split"] == "test").to_numpy()
    grid = dict(model_grid())
    print(f"\n{'='*72}\n{family}: {BLOCK} + {MODEL}\n{'='*72}", flush=True)

    clf = grid[MODEL]()
    clf.fit(X[tr], y[tr])
    p_va, y_va = proba(clf, X[va]), y[va]

    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    thr_f1 = float(np.clip(thr[int(np.argmax(f1[:-1]))], 0.0, 1.0)) if len(thr) else 0.5
    print(f"F1-optimal threshold on validation: {thr_f1:.4f} (best val F1 {f1[:-1].max():.4f})")

    out = []
    for t, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
        out.append(confusion_report(f"{family} VALIDATION (full) {tag}", y_va, p_va, t))
        out[-1].update(family=family, split="val", subset="full")
        show(out[-1])

    # test: refit on train+val, thresholds fixed from validation
    clf_f = grid[MODEL]()
    clf_f.fit(X[tr | va], y[tr | va])
    p_te, y_te = proba(clf_f, X[te]), y[te]
    for t, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
        out.append(confusion_report(f"{family} TEST (full) {tag}", y_te, p_te, t))
        out[-1].update(family=family, split="test", subset="full")
        show(out[-1])

    os.makedirs(EQUAL_DIR, exist_ok=True)
    for name, mask, yy, pp in (("val", va, y_va, p_va), ("test", te, y_te, p_te)):
        if yy.sum() < MIN_POS_FOR_BALANCED:
            print(f"  [skip] {name}-equal for {family}: only {int(yy.sum())} positives "
                  f"(< {MIN_POS_FOR_BALANCED}), a 50/50 subset would be noise")
            continue
        idx = np.flatnonzero(mask)
        sel = balanced_subset(ids[idx], yy, SEED)
        keep = idx[sel]
        pd.DataFrame({"id": ids[keep], "produced": y[keep].astype(bool),
                      "source_split": name, "family": family}).to_csv(
            os.path.join(EQUAL_DIR, f"{family}_{name}_equal.csv"), index=False)
        for t, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
            out.append(confusion_report(f"{family} {name.upper()}-EQUAL 50/50 {tag}",
                                        yy[sel], pp[sel], t))
            out[-1].update(family=family, split=name, subset="equal")
            show(out[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--all-folds", action="store_true",
                    help="include the secondary (leaky) folds too")
    args = ap.parse_args()
    os.makedirs(FAM_DIR, exist_ok=True)

    fs = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits.csv"))
    summ = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))
    if args.families:
        fams = args.families
    else:
        sel = summ if args.all_folds else summ[summ["primary"]]
        fams = list(sel.sort_values("n_test", ascending=False)["family"])
    print(f"{len(fams)} folds: {fams}", flush=True)

    df, emb = load()
    rows = []
    for family in fams:
        rows += run_fold(df, emb, fs, family)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(FAM_DIR, "confusion_tables.csv"), index=False)
    with open(os.path.join(FAM_DIR, "confusion_tables.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.join(FAM_DIR, 'confusion_tables.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
