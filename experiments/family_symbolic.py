#!/usr/bin/env python
"""The contributed `symbolic` classifier on the leave-one-family-out folds.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/family_symbolic.py --shard 0 --num-shards 6
    sbatch experiments/family_symbolic.sbatch

Same protocol as run_symbolic.py --mode final: fit on TRAIN, score VAL (and take
the F1-optimal decision threshold there), then refit on TRAIN+VAL and score TEST
once. Feature extraction and fitting are the contributed
`symbolic_nrps_logistic.py`, imported unmodified.

DEVIATION FROM REPORT.md, deliberate and agreed
-----------------------------------------------
REPORT.md swept 37 configurations of this model (three stages) and selected
n_kmers=150000, l2=3.0, max_iter=2500 on the random split's validation set.
Repeating that sweep on 29 folds would be 29 x 37 slow sparse fits. The selected
configuration is therefore **frozen and refit per fold, with no per-fold sweep**.

Consequence: `symbolic` gets the "frozen" treatment only. In the family_eval.py
table it has no "tuned" arm, so it is comparable to the other methods' frozen
arm and NOT to their tuned arm. If symbolic underperforms on a fold, a stale
hyperparameter cannot be ruled out as the cause.

Why the input CSVs are rebuilt here
-----------------------------------
run_symbolic.py reads $GO_SCRATCH/bgc_clf/symbolic/{train,val,trainval,test}.csv,
but NO SCRIPT IN THIS REPO WRITES THOSE FILES -- they exist on scratch from a
manual step that was never committed. Rather than depend on them, this script
rebuilds equivalent frames in memory from dataset_split.parquet, which is the
same source (`id`, `protein`, `produced`) and matches the observed header
`id,sequence,produced`. The originals are left untouched.

Outputs, under results/family/:
  <fold>_symbolic.json    val + test AP / ROC-AUC and threshold metrics
  <fold>_symbolic.npz     val and test decision scores, with ids
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_symbolic import metrics_at  # noqa: E402
from symbolic_nrps_logistic import (HASH_DIMENSION, SymbolicLogisticModel,  # noqa: E402
                                    hashed_kmer_counts)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(REPO, "experiments", "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "family")
SCRATCH = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")

# REPORT.md's stage-3 winner, frozen. See the deviation note above.
CFG = {"n_kmers": 150000, "l2": 3.0, "min_df": 5, "max_iter": 2500}


def rows_for(d: pd.DataFrame, mask: np.ndarray) -> list[dict[str, str]]:
    """The row dicts symbolic_nrps_logistic expects: id, sequence, produced."""
    sub = d.loc[mask]
    return [{"id": str(i), "sequence": s, "produced": "true" if p else "false"}
            for i, s, p in zip(sub["id"], sub["protein"], sub["produced"])]


def eligible_bins(rows: list[dict[str, str]], min_df: int) -> int:
    """How many hashed 6-mer bins clear `min_df` on this training set.

    SymbolicLogisticModel.fit RAISES if `number_of_kmers` exceeds this count, and
    REPORT.md's frozen 150,000 does not fit every fold: the folds that hold out a
    large family have a smaller, less diverse training set and cannot supply that
    many bins. GxpS is the extreme case -- it is the biggest held-out family, so
    it leaves the smallest training set (7,655 constructs).

    The count is label-blind. It mirrors the `total >= minimum_document_frequency`
    term inside fit(), which sums document frequencies over both classes; the
    class-discriminative ranking that fit() applies afterwards is what uses
    labels, and that is unchanged and still training-only.
    """
    total = np.zeros(HASH_DIMENSION, dtype=np.int32)
    for r in rows:
        bins, _ = hashed_kmer_counts(r["sequence"])
        total[bins] += 1
    return int((total >= min_df).sum())


def run_fold(d_all: pd.DataFrame, fs: pd.DataFrame, family: str) -> None:
    t0 = time.time()
    fold = fs[fs["family"] == family]
    d = d_all.copy()
    d["split"] = d["id"].map(dict(zip(fold["id"], fold["split"])))
    assert d["split"].notna().all(), f"{family}: fold does not cover every id"

    tr = (d["split"] == "train").to_numpy()
    va = (d["split"] == "val").to_numpy()
    te = (d["split"] == "test").to_numpy()
    y_val = d.loc[va, "produced"].to_numpy().astype(int)
    y_te = d.loc[te, "produced"].to_numpy().astype(int)
    print(f"\n=== {family} symbolic: train {tr.sum()} val {va.sum()} test {te.sum()}",
          flush=True)

    tr_rows = rows_for(d, tr)
    avail = eligible_bins(tr_rows, CFG["min_df"])
    n_kmers = min(CFG["n_kmers"], avail)
    if n_kmers < CFG["n_kmers"]:
        print(f"  [cap] only {avail} motif bins clear min_df={CFG['min_df']} on this "
              f"fold's training set; n_kmers {CFG['n_kmers']} -> {n_kmers}", flush=True)

    # fit on TRAIN, score VAL, take the F1-optimal threshold there
    model, _ = SymbolicLogisticModel.fit(
        tr_rows, number_of_kmers=n_kmers,
        minimum_document_frequency=CFG["min_df"], l2=CFG["l2"],
        maximum_iterations=CFG["max_iter"], threshold=0.0)
    s_val = model.decision_function(d.loc[va, "protein"].tolist())
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    thr_f1 = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.0
    val = {"ap": float(average_precision_score(y_val, s_val)),
           "roc_auc": float(roc_auc_score(y_val, s_val)),
           "at_0": metrics_at(y_val, s_val, 0.0),
           "at_f1": metrics_at(y_val, s_val, thr_f1)}
    print(f"  VAL  AP {val['ap']:.4f} ROC-AUC {val['roc_auc']:.4f} "
          f"(F1-optimal threshold {thr_f1:.4f})", flush=True)

    # refit on TRAIN+VAL, score TEST once
    m2, tm = SymbolicLogisticModel.fit(
        rows_for(d, tr | va), number_of_kmers=n_kmers,
        minimum_document_frequency=CFG["min_df"], l2=CFG["l2"],
        maximum_iterations=CFG["max_iter"], threshold=0.0)
    s_te = m2.decision_function(d.loc[te, "protein"].tolist())
    test = {"ap": float(average_precision_score(y_te, s_te)),
            "roc_auc": float(roc_auc_score(y_te, s_te)),
            "converged": bool(tm["optimization"]["success"]),
            "chance_ap": float(y_te.mean()),
            "at_0": metrics_at(y_te, s_te, 0.0),
            "at_f1": metrics_at(y_te, s_te, thr_f1)}
    print(f"  TEST AP {test['ap']:.4f} ROC-AUC {test['roc_auc']:.4f} "
          f"(chance {test['chance_ap']:.4f}) converged={test['converged']} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    with open(os.path.join(OUT_DIR, f"{family}_symbolic.json"), "w") as fh:
        json.dump({"family": family, "config": dict(CFG, n_kmers=n_kmers),
                   "n_kmers_requested": CFG["n_kmers"], "n_kmers_eligible": avail,
                   "n_kmers_capped": bool(n_kmers < CFG["n_kmers"]),
                   "tuning": "frozen from REPORT.md", "val": val, "test": test},
                  fh, indent=2)
    np.savez(os.path.join(OUT_DIR, f"{family}_symbolic.npz"),
             val_ids=d.loc[va, "id"].to_numpy(), val_scores=s_val, val_y=y_val,
             test_ids=d.loc[te, "id"].to_numpy(), test_scores=s_te, test_y=y_te)


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
                if not os.path.exists(os.path.join(OUT_DIR, f"{f}_symbolic.json"))]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} folds -> {todo}", flush=True)
    if not todo:
        return 0

    d_all = pd.read_parquet(os.path.join(SCRATCH, "dataset_split.parquet"))
    failed = []
    for family in todo:
        try:
            run_fold(d_all, fs, family)
        except Exception as e:
            # One bad fold must not kill the shard. The first symbolic run lost 5
            # folds this way: GxpS and XabS raised on n_kmers, and because each was
            # position 0 of its shard, the 5 folds queued behind them were never
            # attempted. Failures are reported at the end instead.
            print(f"!!! {family} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(family)
    if failed:
        print(f"\n!!! {len(failed)} fold(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
