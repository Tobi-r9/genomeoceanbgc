#!/usr/bin/env python
"""REPORT.md's evaluation, re-run on the leave-one-family-out folds.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/family_eval.py --shard 0 --num-shards 6
    sbatch experiments/family_eval.sbatch

For every fold in `splits/family_splits.csv` this reproduces the protocol of
[REPORT.md](REPORT.md) for the three cheap methods:

    emb_mean         GenomeOcean-4B mean-pooled embedding (3072-d)
    modules+bigram   which modules are present + which adjacent pairs
    modules          which modules are present, order-blind

`symbolic` is handled separately by family_symbolic.py -- it needs protein
sequences and its own fitting code.

Protocol per fold, matching train_classifiers.py + test_table.py:
  1. sweep all 19 classifier configurations per block, fit on TRAIN, score VAL;
  2. **tuned** arm -- take that fold's own validation winner per block;
     **frozen** arm -- take the configuration REPORT.md selected on the random
     split, refit unchanged. Both are reported, so a collapse caused by the
     hard split can be told apart from one caused by a stale hyperparameter;
  3. refit the chosen config on TRAIN+VAL and score TEST exactly once;
  4. percentile bootstrap CI on test AP, and a paired bootstrap of each baseline
     against emb_mean on the same resampled rows.

Nothing here modifies the existing pipeline. train_classifiers.py, analyze_val.py
and test_table.py are imported read-only and are unchanged, so the original
random-split runs stay reproducible.

Outputs (all under results/family/, nothing overwrites a random-split result):
  <fold>_val_sweep.csv   every block x model validation score for that fold
  <fold>_test.csv        one row per (arm, method) with test AP, CI, ROC-AUC
  <fold>_paired.csv      paired bootstrap of each baseline vs emb_mean on test
  <fold>_scores.npz      test scores per method, for later re-analysis
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
from analyze_val import paired_bootstrap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(RESULT_DIR)
OUT_DIR = os.path.join(RESULTS_ROOT, "family")

BLOCKS = ["emb_mean", "modules+bigram", "modules"]

# What REPORT.md's final test table used, chosen on the RANDOM split's
# validation set. Carried over unchanged as the "frozen" arm.
FROZEN = {
    "emb_mean": "logreg_C10.0",
    "modules+bigram": "hgb_lr0.1_l63",
    "modules": "hgb_lr0.1_l63",
}
N_BOOT = 2000


def fold_frame(df: pd.DataFrame, fs: pd.DataFrame, family: str) -> pd.DataFrame:
    """df with `split` replaced by this family's fold. Never mutates the input."""
    fold = fs[fs["family"] == family]
    out = df.copy()
    out["split"] = out["id"].map(dict(zip(fold["id"], fold["split"])))
    assert out["split"].notna().all(), f"{family}: fold does not cover every id"
    return out


def run_fold(df: pd.DataFrame, emb: dict, fs: pd.DataFrame, family: str) -> None:
    t_fold = time.time()
    d = fold_frame(df, fs, family)
    # feature_blocks derives the module-bigram vocabulary from the TRAIN rows,
    # so it must see the fold's split, not the random one. Pairs unseen in this
    # fold's training set are dropped rather than leaking their existence.
    blocks = feature_blocks(d, emb)
    grid = dict(model_grid())

    tr = (d["split"] == "train").to_numpy()
    va = (d["split"] == "val").to_numpy()
    te = (d["split"] == "test").to_numpy()
    tv = tr | va
    y = d["produced"].to_numpy().astype(int)
    yte = y[te]
    print(f"\n=== {family}: train {tr.sum()} val {va.sum()} test {te.sum()} "
          f"| test positive rate {yte.mean():.4f} ({yte.sum()} positives)", flush=True)

    # ---- 1. validation sweep ---------------------------------------------
    sweep = []
    for bname in BLOCKS:
        X = blocks[bname]
        for mname, factory in model_grid():
            t0 = time.time()
            try:
                clf = factory()
                clf.fit(X[tr], y[tr])
                p = proba(clf, X[va])
                sweep.append({"family": family, "block": bname, "model": mname,
                              "n_features": X.shape[1],
                              "val_ap": float(average_precision_score(y[va], p)),
                              "val_roc_auc": float(roc_auc_score(y[va], p)),
                              "fit_s": round(time.time() - t0, 1)})
            except Exception as e:                      # keep the sweep going
                print(f"  {bname:16s} {mname:20s} FAILED {type(e).__name__}: {e}",
                      flush=True)
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(OUT_DIR, f"{family}_val_sweep.csv"), index=False)

    # ---- 2-3. tuned and frozen arms, one test evaluation each -------------
    rows, preds = [], {}
    for arm in ("tuned", "frozen"):
        for bname in BLOCKS:
            sub = sw[sw["block"] == bname]
            if sub.empty:
                continue
            if arm == "tuned":
                pick = sub.loc[sub["val_ap"].idxmax()]
                mname, val_ap = pick["model"], float(pick["val_ap"])
            else:
                mname = FROZEN[bname]
                hit = sub[sub["model"] == mname]
                val_ap = float(hit["val_ap"].iloc[0]) if len(hit) else np.nan
            clf = grid[mname]()
            clf.fit(blocks[bname][tv], y[tv])          # refit on train+val
            p = proba(clf, blocks[bname][te])          # scored ONCE
            lo, hi = bootstrap_ap(yte, p, n_boot=N_BOOT)
            preds[(arm, bname)] = p
            rows.append({"family": family, "arm": arm, "method": bname,
                         "model": mname, "val_ap": val_ap,
                         "test_ap": float(average_precision_score(yte, p)),
                         "ap_lo": lo, "ap_hi": hi,
                         "test_roc_auc": float(roc_auc_score(yte, p)),
                         "chance_ap": float(yte.mean()),
                         "n_test": int(te.sum()), "test_positives": int(yte.sum())})
            print(f"  [{arm:6s}] {bname:16s} {mname:20s} "
                  f"test AP {rows[-1]['test_ap']:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"ROC-AUC {rows[-1]['test_roc_auc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{family}_test.csv"), index=False)

    # ---- 4. paired bootstrap against the embedding -----------------------
    pair = []
    for arm in ("tuned", "frozen"):
        if (arm, "emb_mean") not in preds:
            continue
        for bname in BLOCKS:
            if bname == "emb_mean" or (arm, bname) not in preds:
                continue
            m, lo, hi, pw = paired_bootstrap(yte, preds[(arm, "emb_mean")],
                                             preds[(arm, bname)], n_boot=N_BOOT)
            pair.append({"family": family, "arm": arm, "vs": bname,
                         "d_ap": m, "lo": lo, "hi": hi, "p_emb_better": pw,
                         "significant": bool(lo > 0)})
            print(f"  [{arm:6s}] emb_mean - {bname:16s} dAP {m:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] P={pw:.3f}", flush=True)
    pd.DataFrame(pair).to_csv(os.path.join(OUT_DIR, f"{family}_paired.csv"), index=False)

    np.savez(os.path.join(OUT_DIR, f"{family}_scores.npz"),
             ids=d.loc[te, "id"].to_numpy(), y=yte,
             **{f"{a}__{b}": p for (a, b), p in preds.items()})
    print(f"  {family} done in {(time.time()-t_fold)/60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--families", nargs="*", default=None,
                    help="explicit family list; overrides sharding")
    ap.add_argument("--redo", action="store_true",
                    help="recompute folds whose output already exists")
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
                if not os.path.exists(os.path.join(OUT_DIR, f"{f}_test.csv"))]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} folds -> {todo}", flush=True)
    if not todo:
        return 0

    df, emb = load()
    failed = []
    for family in todo:
        try:
            run_fold(df, emb, fs, family)
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
