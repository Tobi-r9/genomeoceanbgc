#!/usr/bin/env python
"""Confusion tables, and balanced 50/50 evaluation subsets, for val and test.

    python experiments/confusion_and_balanced.py            # validation only
    python experiments/confusion_and_balanced.py --test      # + TEST (one shot)

Model is the one committed in REPORT.md before any test data was touched:
`emb_mean` + logistic regression C=10, class_weight="balanced".

Two fits, matching the protocol already documented:
  * validation numbers come from the model fit on TRAIN only;
  * test numbers come from the model refit on TRAIN+VAL, standard once model
    selection is frozen.

Thresholds. A confusion table needs one, and choosing it is a model decision, so
it is chosen on VALIDATION and then applied unchanged to test. Two operating
points are reported: 0.5 (the natural point for a class-weighted model) and the
F1-optimal threshold on validation.

Balanced subsets keep every positive and randomly subsample negatives to match
(seeded). Note what this does and does not change: recall and ROC-AUC are
prevalence-independent, so they carry over; precision, AP and accuracy are NOT --
on a 50/50 set they look much better simply because positives are 3.5x more
common than in reality. The balanced set makes accuracy interpretable; it does not
make the task easier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import RESULT_DIR, SEED, feature_blocks, load, model_grid, proba  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
BLOCK, MODEL = "emb_mean", "logreg_C10.0"


def confusion_report(name, y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    yhat = (p >= thr).astype(int)
    d = {
        "set": name, "threshold": round(float(thr), 4),
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }
    return d


def show(d):
    print(f"\n=== {d['set']}  (threshold {d['threshold']}) ===")
    print(f"  n={d['n']}  positives={d['n_pos']}  negatives={d['n_neg']}  "
          f"positive rate={d['positive_rate']:.4f}")
    print( "                      predicted")
    print( "                  not prod.   produced")
    print(f"  actual not prod.  {d['TN']:8d}   {d['FP']:8d}")
    print(f"  actual produced   {d['FN']:8d}   {d['TP']:8d}")
    print(f"  accuracy {d['accuracy']:.4f} | balanced acc {d['balanced_accuracy']:.4f} | "
          f"MCC {d['mcc']:.4f}")
    print(f"  precision {d['precision']:.4f} | recall {d['recall']:.4f} | "
          f"specificity {d['specificity']:.4f} | F1 {d['f1']:.4f}")
    print(f"  AP {d['ap']:.4f} | ROC-AUC {d['roc_auc']:.4f}  "
          f"(chance AP = {d['positive_rate']:.4f})")


def balanced_subset(ids, y, seed):
    """All positives + an equal number of randomly chosen negatives."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    take = rng.choice(neg, size=len(pos), replace=False)
    sel = np.sort(np.concatenate([pos, take]))
    assert y[sel].sum() * 2 == len(sel), "subset is not exactly 50/50"
    return sel


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--test", action="store_true",
                     help="also evaluate the TEST set (consumes the held-out split)")
    args = ap_.parse_args()

    df, emb = load()
    X = feature_blocks(df, emb)[BLOCK]
    y = df["produced"].to_numpy().astype(int)
    ids = df["id"].to_numpy()
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()

    grid = dict(model_grid())
    print(f"model: {BLOCK} + {MODEL}")
    # The train+val refit emits a ConvergenceWarning (lbfgs hits max_iter=3000).
    # Checked and immaterial: with max_iter=20000 it converges at 3077 iterations
    # and gives test AP 0.7331 vs 0.7335, i.e. a single sample moves in the
    # confusion matrix. max_iter is left at the value that was committed before
    # test data was touched rather than tuned after the fact.

    # ---- validation (model fit on train only) ------------------------------
    clf = grid[MODEL]()
    clf.fit(X[tr], y[tr])
    p_va = proba(clf, X[va])
    y_va = y[va]

    # F1-optimal threshold, chosen on VALIDATION and reused for test.
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    thr_f1 = float(thr[max(0, int(np.argmax(f1)) - 0)]) if len(thr) else 0.5
    thr_f1 = float(np.clip(thr_f1, 0.0, 1.0))
    print(f"F1-optimal threshold on validation: {thr_f1:.4f} "
          f"(best val F1 {f1.max():.4f})")

    out = []
    for thr_val, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
        out.append(confusion_report(f"VALIDATION (full) {tag}", y_va, p_va, thr_val))
        show(out[-1])

    # ---- val-equal: exactly 50/50 -----------------------------------------
    va_idx = np.flatnonzero(va)
    sel = balanced_subset(ids[va_idx], y_va, SEED)
    ve = va_idx[sel]
    pd.DataFrame({"id": ids[ve], "produced": y[ve].astype(bool),
                  "source_split": "val"}).to_csv(
        os.path.join(SPLIT_DIR, "val_equal.csv"), index=False)
    for thr_val, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
        out.append(confusion_report(f"VAL-EQUAL 50/50 {tag}", y_va[sel], p_va[sel], thr_val))
        show(out[-1])
    print(f"\nwrote {os.path.join(SPLIT_DIR, 'val_equal.csv')} "
          f"({len(ve)} rows, {int(y[ve].sum())} positives)")

    # ---- test -------------------------------------------------------------
    if args.test:
        print("\n" + "=" * 72)
        print("TEST EVALUATION -- refit on train+val, thresholds fixed from validation")
        print("=" * 72)
        fit = tr | va
        clf_f = grid[MODEL]()
        clf_f.fit(X[fit], y[fit])
        p_te = proba(clf_f, X[te])
        y_te = y[te]
        for thr_val, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
            out.append(confusion_report(f"TEST (full) {tag}", y_te, p_te, thr_val))
            show(out[-1])

        te_idx = np.flatnonzero(te)
        sel_t = balanced_subset(ids[te_idx], y_te, SEED)
        tequal = te_idx[sel_t]
        pd.DataFrame({"id": ids[tequal], "produced": y[tequal].astype(bool),
                      "source_split": "test"}).to_csv(
            os.path.join(SPLIT_DIR, "test_equal.csv"), index=False)
        for thr_val, tag in ((0.5, "@0.5"), (thr_f1, "@valF1")):
            out.append(confusion_report(f"TEST-EQUAL 50/50 {tag}", y_te[sel_t],
                                        p_te[sel_t], thr_val))
            show(out[-1])
        print(f"\nwrote {os.path.join(SPLIT_DIR, 'test_equal.csv')} "
              f"({len(tequal)} rows, {int(y[tequal].sum())} positives)")

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(RESULT_DIR, "confusion_tables.csv"), index=False)
    with open(os.path.join(RESULT_DIR, "confusion_tables.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(RESULT_DIR, 'confusion_tables.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
