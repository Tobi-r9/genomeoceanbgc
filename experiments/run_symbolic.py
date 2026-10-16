#!/usr/bin/env python
"""Run the symbolic NRPS classifier on our split, with our metrics.

    python experiments/run_symbolic.py --mode sweep --shard 0 --num-shards 6
    python experiments/run_symbolic.py --mode final --n-kmers 8000 --l2 1.0 --max-iter 600

`symbolic_nrps_logistic.py` (contributed, unmodified) reports only
threshold-based metrics -- accuracy, precision, recall, F1 -- and its own
stratified split. This wrapper instead:

  * uses OUR train/val/test split;
  * feeds it the PROTEIN sequence, which is what its features assume (amino-acid
    alphabet, and NRPS domain motifs such as SSGTT / YKTGDL);
  * adds average precision and ROC-AUC from `decision_function`, so the numbers
    are comparable with every other row of the report;
  * tunes on validation only, then evaluates test once, refitting on train+val.

Note on its default `--max-iterations 180`: at that setting L-BFGS terminates on
the iteration cap with `success: false`, so the default is an under-fit model
rather than a converged one. Iteration count is therefore swept as a real
hyperparameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbolic_nrps_logistic import SymbolicLogisticModel, labels, read_csv  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO, "experiments", "results")
DATA = Path(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean")) \
    / "bgc_clf" / "symbolic"

GRID1 = [
    {"n_kmers": nk, "l2": l2, "min_df": 5, "max_iter": mi}
    for nk in (2000, 8000, 20000)
    for l2 in (0.3, 1.0, 3.0, 10.0)
    for mi in (180, 600)
]

# Stage 2. In stage 1 L-BFGS never converged -- every configuration stopped on
# the iteration cap -- and val AP was still climbing with the budget (mean 0.659
# at 180 iterations, 0.668 at 600). Reporting a model that is merely under-fit
# would understate the method, so the budget is pushed until it either converges
# or plateaus, around the best stage-1 region (large n_kmers, moderate L2).
GRID2 = [
    {"n_kmers": nk, "l2": l2, "min_df": 5, "max_iter": mi}
    for nk in (8000, 20000, 40000)
    for l2 in (1.0, 3.0, 10.0)
    for mi in (2500,)
]

# Stage 3. Stage 2 selected n_kmers=40000, the top of its range, so the optimum
# may have been clipped. Push the motif budget further to find the interior
# maximum (same reason the logistic-regression C grid was extended earlier).
GRID3 = [
    {"n_kmers": nk, "l2": l2, "min_df": 5, "max_iter": 2500}
    for nk in (40000, 80000, 150000)
    for l2 in (3.0, 10.0)
]

GRIDS = {"stage1": GRID1, "stage2": GRID2, "stage3": GRID3}


def load_split(name):
    rows = read_csv(DATA / f"{name}.csv", require_labels=True)
    return rows, [r["sequence"] for r in rows], labels(rows).astype(int)


def fit_score(train_rows, eval_seqs, cfg):
    t0 = time.time()
    model, train_metrics = SymbolicLogisticModel.fit(
        train_rows, number_of_kmers=cfg["n_kmers"],
        minimum_document_frequency=cfg["min_df"], l2=cfg["l2"],
        maximum_iterations=cfg["max_iter"], threshold=0.0)
    scores = model.decision_function(eval_seqs)
    return model, scores, train_metrics, time.time() - t0


def metrics_at(y, s, thr):
    tn, fp, fn, tp = confusion_matrix(y, (s >= thr).astype(int), labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(thr), "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "accuracy": float((tp + tn) / len(y)),
            "balanced_accuracy": float(0.5 * (rec + tn / (tn + fp) if tn + fp else 0)),
            "precision": float(prec), "recall": float(rec),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sweep", "final"), default="sweep")
    ap.add_argument("--grid", choices=tuple(GRIDS), default="stage1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-kmers", type=int, default=8000)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.mode == "sweep":
        train_rows, _, _ = load_split("train")
        _, val_seqs, y_val = load_split("val")
        todo = GRIDS[args.grid][args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(todo)} configs", flush=True)
        out = []
        for cfg in todo:
            _, s, tm, secs = fit_score(train_rows, val_seqs, cfg)
            row = dict(cfg,
                       val_ap=float(average_precision_score(y_val, s)),
                       val_roc_auc=float(roc_auc_score(y_val, s)),
                       converged=bool(tm["optimization"]["success"]),
                       n_iter=int(tm["optimization"]["iterations"]),
                       fit_s=round(secs, 1))
            out.append(row)
            print(f"  n_kmers={cfg['n_kmers']:6d} l2={cfg['l2']:5.1f} "
                  f"max_iter={cfg['max_iter']:4d} -> val AP {row['val_ap']:.4f} "
                  f"ROC-AUC {row['val_roc_auc']:.4f} converged={row['converged']} "
                  f"({secs:.0f}s)", flush=True)
        p = os.path.join(RESULT_DIR, f"symbolic_sweep_{args.grid}_{args.shard}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"wrote {p}")
        return 0

    # ---- final: refit on train+val, evaluate test once ---------------------
    cfg = {"n_kmers": args.n_kmers, "l2": args.l2,
           "min_df": args.min_df, "max_iter": args.max_iter}
    print(f"FINAL config: {cfg}")

    # Threshold is chosen on validation (model fit on train only), then applied
    # unchanged to test -- same protocol as every other model in the report.
    train_rows, _, _ = load_split("train")
    _, val_seqs, y_val = load_split("val")
    _, s_val, _, _ = fit_score(train_rows, val_seqs, cfg)
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    thr_f1 = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.0
    val = {"ap": float(average_precision_score(y_val, s_val)),
           "roc_auc": float(roc_auc_score(y_val, s_val)),
           "at_0": metrics_at(y_val, s_val, 0.0),
           "at_f1": metrics_at(y_val, s_val, thr_f1)}
    print(f"VAL  AP {val['ap']:.4f}  ROC-AUC {val['roc_auc']:.4f}  "
          f"(F1-optimal decision threshold {thr_f1:.4f})")

    tv_rows, _, _ = load_split("trainval")
    _, test_seqs, y_te = load_split("test")
    _, s_te, tm, _ = fit_score(tv_rows, test_seqs, cfg)
    test = {"ap": float(average_precision_score(y_te, s_te)),
            "roc_auc": float(roc_auc_score(y_te, s_te)),
            "converged": bool(tm["optimization"]["success"]),
            "at_0": metrics_at(y_te, s_te, 0.0),
            "at_f1": metrics_at(y_te, s_te, thr_f1)}
    print(f"TEST AP {test['ap']:.4f}  ROC-AUC {test['roc_auc']:.4f}")
    for tag in ("at_0", "at_f1"):
        m = test[tag]
        print(f"  test {tag} thr={m['threshold']:.3f}: "
              f"TN {m['TN']} FP {m['FP']} FN {m['FN']} TP {m['TP']} | "
              f"acc {m['accuracy']:.4f} prec {m['precision']:.4f} "
              f"rec {m['recall']:.4f} F1 {m['f1']:.4f}")

    p = os.path.join(RESULT_DIR, "symbolic_final.json")
    with open(p, "w") as fh:
        json.dump({"config": cfg, "val": val, "test": test}, fh, indent=2)
    print(f"wrote {p}")
    # Scores kept so the paired bootstrap against the other models can reuse them.
    np.savez(os.path.join(RESULT_DIR, "symbolic_scores.npz"),
             val_scores=s_val, val_y=y_val, test_scores=s_te, test_y=y_te)
    return 0


if __name__ == "__main__":
    sys.exit(main())
