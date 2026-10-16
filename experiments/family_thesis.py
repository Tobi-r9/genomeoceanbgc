#!/usr/bin/env python
"""The NRPS-Models-and-data thesis method, on the leave-one-family-out folds.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/family_thesis.py --shard 0 --num-shards 6
    sbatch experiments/family_thesis.sbatch

Reimplements the pipeline of `NRPS-Models-and-data/Recent_Models/model.py` (the
master-thesis model) against our dataset and our family folds, so it can sit in
the same table as emb_mean / symbolic / modules+bigram / modules.

Their feature set, reproduced:
  * organism one-hot, one column per organism seen in the FITTING set, set by a
    SUBSTRING test on the construct's comma-joined organism string;
  * architecture compatibility along the chain -- for each adjacent pair,
    downstream_Architecture(i) == module_architecture(i+1) and
    upstream_Architecture(i+1) == module_architecture(i) -- reduced to
    upstream_sum / downstream_sum / upstream_len / downstream_len;
  * GC content per module, capped at 65 %, reduced to GC_mean / GC_median /
    GC_max / GC_count;
  * TF-IDF over character 3-5-grams of the concatenated nucleotide sequence and
    of the concatenated protein, 1000 features each.
Then StandardScaler and XGBoost with their exact parameters (binary:logistic,
max_depth 6, eta 0.05, subsample 0.9, colsample_bytree 0.9, hist, 300 rounds,
scale_pos_weight = negatives/positives).

Four deviations, all deliberate and all documented in FAMILY_RESULTS.md:

1. **TF-IDF, the scaler and the organism vocabulary are fitted on the FITTING
   SET ONLY.** model.py fits the vectorizers on the training block but the
   sibling notebooks fit them on the whole dataset before splitting, which
   leaks. Fitting on train only is the non-leaky reading and is what every other
   method in this report does.
2. **Our protocol, not theirs.** model.py has no validation set (it splits
   train/test by row position). Here: fit on TRAIN and score VAL, then refit on
   TRAIN+VAL and score TEST once -- the protocol used by every other method here.
3. **No hyperparameter sweep**, because their code has none. Like `symbolic`,
   this method therefore has a frozen arm only and is comparable to the other
   methods' frozen arm.
4. **`module_architecture` is the architecture column.** Their donor sheet had a
   single `Module_Architecture`; ours has both `module_architecture` (16 values)
   and `Module_Architecture_uncommon` (44). Measured on all 71,159 adjacent
   module pairs, the match feature fires for 42.4 % / 42.9 % of pairs with
   `module_architecture` but only 0.2 % / 2.0 % with the `_uncommon` column, so
   the latter would have been a silently dead feature.

The substring organism test is reproduced as written, not fixed: `Cystobacter
sp.` matches `Cystobacter sp. SBCb004`. It is their method.

Outputs, under results/family/:
  <fold>_thesis.json   val + test AP / ROC-AUC, feature-block sizes
  <fold>_thesis.npz    val and test scores, with ids
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS_ROOT = os.path.join(REPO, "experiments", "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "family")
SCRATCH = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")
DONORS = os.path.join(REPO, "dataset", "donors_data.xlsx")

PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 6,
          "eta": 0.05, "subsample": 0.9, "colsample_bytree": 0.9,
          "tree_method": "hist", "random_state": 42}
N_ROUNDS = 300
GC_CAP = 65


def gc_content(seq: str) -> float:
    if not isinstance(seq, str) or not seq:
        return 0.0
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) * 100


def donor_maps():
    don = pd.read_excel(DONORS)
    don["module number"] = don["module number"].astype(str).str.strip()
    org = dict(zip(don["module number"], don["organism"].astype(str)))
    gc = {m: gc_content(s) for m, s in zip(don["module number"], don["sequence_nt"])}
    arch = (don.groupby("module number")[["module_architecture",
                                          "downstream_Architecture",
                                          "upstream_Architecture"]]
            .first().to_dict("index"))
    return org, gc, arch


def static_features(mod_lists, org, gc, arch):
    """Everything except TF-IDF and the organism one-hot (which need a vocabulary)."""
    organisms, num = [], np.zeros((len(mod_lists), 8), dtype=float)
    for i, l in enumerate(mod_lists):
        organisms.append(",".join(str(org.get(m, "Unknown")) for m in l))
        dm, um = [], []
        for j in range(len(l) - 1):
            a, b = arch.get(l[j], {}), arch.get(l[j + 1], {})
            dm.append(int(a.get("downstream_Architecture") == b.get("module_architecture")))
            um.append(int(b.get("upstream_Architecture") == a.get("module_architecture")))
        g = [min(gc.get(m, 0.0), GC_CAP) for m in l]
        num[i] = [sum(um), sum(dm), len(um), len(dm),
                  np.mean(g) if g else 0, np.median(g) if g else 0,
                  np.max(g) if g else 0, len(g)]
    return organisms, num


def build(fit_idx, eval_idx, organisms, num, nt, aa):
    """Fit every transform on fit_idx only, then apply to both blocks."""
    vocab = pd.Series([organisms[i] for i in fit_idx]).str.split(",").explode().str.strip()
    vocab = [o for o in vocab.dropna().unique()]
    def onehot(idx):
        return np.array([[1 if o in organisms[i] else 0 for o in vocab] for i in idx],
                        dtype=float)
    v_nt = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=1000)
    v_aa = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=1000)
    Xnt_f = v_nt.fit_transform([nt[i] for i in fit_idx]).toarray()
    Xaa_f = v_aa.fit_transform([aa[i] for i in fit_idx]).toarray()
    Xnt_e = v_nt.transform([nt[i] for i in eval_idx]).toarray()
    Xaa_e = v_aa.transform([aa[i] for i in eval_idx]).toarray()
    Xf = np.hstack([onehot(fit_idx), num[fit_idx], Xnt_f, Xaa_f])
    Xe = np.hstack([onehot(eval_idx), num[eval_idx], Xnt_e, Xaa_e])
    sc = StandardScaler().fit(Xf)
    return sc.transform(Xf), sc.transform(Xe), len(vocab)


def fit_score(fit_idx, eval_idx, y, organisms, num, nt, aa):
    Xf, Xe, n_org = build(fit_idx, eval_idx, organisms, num, nt, aa)
    yf = y[fit_idx]
    p = dict(PARAMS, scale_pos_weight=(len(yf) - yf.sum()) / max(yf.sum(), 1))
    bst = xgb.train(p, xgb.DMatrix(Xf, label=yf), num_boost_round=N_ROUNDS)
    return bst.predict(xgb.DMatrix(Xe)), Xf.shape[1], n_org


def run_fold(d, fs, family, org, gc, arch, mod_lists, nt, aa):
    t0 = time.time()
    fold = fs[fs["family"] == family]
    sp = d["id"].map(dict(zip(fold["id"], fold["split"])))
    assert sp.notna().all(), f"{family}: fold does not cover every id"
    tr = np.where((sp == "train").to_numpy())[0]
    va = np.where((sp == "val").to_numpy())[0]
    te = np.where((sp == "test").to_numpy())[0]
    y = d["produced"].to_numpy().astype(int)
    print(f"\n=== {family} thesis: train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    organisms, num = static_features(mod_lists, org, gc, arch)

    s_val, nfeat, n_org = fit_score(tr, va, y, organisms, num, nt, aa)
    val = {"ap": float(average_precision_score(y[va], s_val)),
           "roc_auc": float(roc_auc_score(y[va], s_val))}
    print(f"  VAL  AP {val['ap']:.4f} ROC-AUC {val['roc_auc']:.4f} "
          f"({nfeat} features, {n_org} organisms)", flush=True)

    s_te, nfeat2, n_org2 = fit_score(np.concatenate([tr, va]), te, y, organisms, num, nt, aa)
    test = {"ap": float(average_precision_score(y[te], s_te)),
            "roc_auc": float(roc_auc_score(y[te], s_te)),
            "chance_ap": float(y[te].mean())}
    print(f"  TEST AP {test['ap']:.4f} ROC-AUC {test['roc_auc']:.4f} "
          f"(chance {test['chance_ap']:.4f}) [{(time.time()-t0)/60:.1f} min]", flush=True)

    with open(os.path.join(OUT_DIR, f"{family}_thesis.json"), "w") as fh:
        json.dump({"family": family, "params": PARAMS, "n_rounds": N_ROUNDS,
                   "n_features_val": nfeat, "n_features_test": nfeat2,
                   "n_organisms_val": n_org, "n_organisms_test": n_org2,
                   "tuning": "none (their code has no sweep)",
                   "val": val, "test": test}, fh, indent=2)
    np.savez(os.path.join(OUT_DIR, f"{family}_thesis.npz"),
             val_ids=d["id"].to_numpy()[va], val_scores=s_val, val_y=y[va],
             test_ids=d["id"].to_numpy()[te], test_scores=s_te, test_y=y[te])


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
                if not os.path.exists(os.path.join(OUT_DIR, f"{f}_thesis.json"))]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} folds -> {todo}", flush=True)
    if not todo:
        return 0

    d = pd.read_parquet(os.path.join(SCRATCH, "dataset_split.parquet"))
    org, gc, arch = donor_maps()
    mod_lists = [[t.strip() for t in str(s).split(",")] for s in d["modules"]]
    nt = d["sequence"].tolist()
    aa = d["protein"].tolist()

    failed = []
    for family in todo:
        try:
            run_fold(d, fs, family, org, gc, arch, mod_lists, nt, aa)
        except Exception as e:
            print(f"!!! {family} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(family)
    if failed:
        print(f"\n!!! {len(failed)} fold(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
