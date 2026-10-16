#!/usr/bin/env python
"""Lightweight classifiers on frozen GenomeOcean-4B-bgcFM embeddings.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/train_classifiers.py                 # tune on train/val
    python experiments/train_classifiers.py --final         # ONE test evaluation

THE TEST SET IS NOT TOUCHED unless --final is passed. Every model-selection
decision here is made on validation only.

Metrics: the task is imbalanced (14.1 % positive), so accuracy is useless -- a
constant "not produced" scores 85.9 %. Primary metric is **average precision**
(area under precision-recall), with ROC-AUC alongside.

Baselines matter more than the models here. Frozen embeddings of a 4 B model are
only interesting if they beat cheap features, so we also fit the same
classifiers on:
  * `len`     : sequence length + module count (3 features) -- pure shortcut signal
  * `aac`     : amino-acid composition (20) + length -- classic protein baseline
  * `modules` : multi-hot over the 1384 donor modules -- the strongest honest
                baseline, since a construct IS its module set. If embeddings do
                not beat this, they add nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")
RESULT_DIR = os.path.join(REPO, "experiments", "results")
SEED = 20260805
AAS = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.read_parquet(os.path.join(SCRATCH, "dataset_split.parquet"))
    emb_dir = os.path.join(SCRATCH, "emb")
    shards = sorted(f for f in os.listdir(emb_dir) if f.endswith(".npz"))
    if not shards:
        raise SystemExit(f"no embedding shards in {emb_dir} -- run experiments/embed.sbatch")
    ids, mean, last, ntok, trunc = [], [], [], [], []
    for f in shards:
        z = np.load(os.path.join(emb_dir, f))
        ids.append(z["ids"]); mean.append(z["mean"]); last.append(z["last"])
        ntok.append(z["n_tokens"]); trunc.append(z["truncated"])
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    ids = ids[order]
    emb = {"mean": np.concatenate(mean)[order], "last": np.concatenate(last)[order]}
    n_tokens = np.concatenate(ntok)[order]
    truncated = np.concatenate(trunc)[order]
    assert len(np.unique(ids)) == len(ids), "duplicate ids across shards"

    df = df.set_index("id").loc[ids].reset_index()
    df["n_tokens"] = n_tokens
    df["truncated"] = truncated
    print(f"loaded {len(df)} rows, {len(shards)} shards, "
          f"embedding dim {emb['mean'].shape[1]}, "
          f"truncated at 10240 tokens: {int(truncated.sum())}")
    return df, emb


def feature_blocks(df: pd.DataFrame, emb: dict) -> dict[str, np.ndarray]:
    n = len(df)
    length = np.stack([df["nt_len"].to_numpy(float),
                       df["aa_len"].to_numpy(float),
                       df["n_modules"].to_numpy(float)], axis=1)

    aac = np.zeros((n, len(AAS)), dtype=np.float32)
    for i, p in enumerate(df["protein"].to_numpy()):
        c = Counter(p)
        tot = max(len(p), 1)
        for j, a in enumerate(AAS):
            aac[i, j] = c.get(a, 0) / tot

    mod_lists = [[m.strip() for m in str(s).split(",")] for s in df["modules"]]
    vocab = sorted({m for ms in mod_lists for m in ms})
    vidx = {m: j for j, m in enumerate(vocab)}
    mods = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, ms in enumerate(mod_lists):
        for m in ms:
            mods[i, vidx[m]] = 1.0

    # Order-aware variant. The plain `mods` multi-hot is a bag: it cannot tell
    # apart two constructs built from the same modules in a different order,
    # and 264 constructs (1.65 %) are exactly such permutation pairs carrying
    # OPPOSITE labels, so a bag-of-modules model provably cannot fit them.
    # Adjacent pairs are the natural fix and match the `Transitions` sheet of
    # experimental_combinatorial_data.xlsx (from_module -> to_module success
    # rates). '^' and '$' sentinels also capture which module is first/last.
    # Vocabulary is built on TRAIN ONLY; pairs unseen in training are ignored
    # at val/test rather than leaking their existence into the features.
    bigrams = [list(zip(["^"] + ms, ms + ["$"])) for ms in mod_lists]
    tr_mask = (df["split"] == "train").to_numpy()
    bvocab = sorted({p for i, ps in enumerate(bigrams) if tr_mask[i] for p in ps})
    bidx = {p: j for j, p in enumerate(bvocab)}
    bg = np.zeros((n, len(bvocab)), dtype=np.float32)
    n_unseen = 0
    for i, ps in enumerate(bigrams):
        for p in ps:
            j = bidx.get(p)
            if j is None:
                n_unseen += 1
            else:
                bg[i, j] = 1.0
    print(f"module bigrams: {len(bvocab)} in train vocab, "
          f"{n_unseen} occurrences unseen in training (ignored)")
    print(f"feature blocks: len {length.shape[1]}, aac {aac.shape[1]}, "
          f"modules {mods.shape[1]}, emb {emb['mean'].shape[1]}")

    # L2-normalising embeddings is a standard trick for linear probes: it removes
    # the magnitude component, which here tracks sequence length.
    mean_l2 = emb["mean"] / np.linalg.norm(emb["mean"], axis=1, keepdims=True)

    return {
        "len": length,
        "aac": np.hstack([aac, length]),
        "modules": mods,
        "modules_bigram": bg,
        "modules+bigram": np.hstack([mods, bg]),
        "emb_mean": emb["mean"],
        "emb_last": emb["last"],
        "emb_mean_l2": mean_l2,
        "emb_mean+len": np.hstack([emb["mean"], length]),
        "emb_mean+aac": np.hstack([emb["mean"], aac, length]),
        "emb_mean+modules": np.hstack([emb["mean"], mods]),
        "emb_mean+modules+aac": np.hstack([emb["mean"], mods, aac, length]),
    }


# ---------------------------------------------------------------------- models
def model_grid(seed=SEED):
    """(name, factory) pairs. Deliberately small and cheap."""
    grid = [("majority", lambda: DummyClassifier(strategy="prior"))]
    # C extends well past 10: the first sweep selected C=10.0, the edge of the
    # original grid, so the optimum could have been clipped.
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0):
        grid.append((f"logreg_C{C}", lambda C=C: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                               random_state=seed))))
    for h in ((64,), (256,), (256, 64)):
        for alpha in (1e-4, 1e-2):
            tag = "x".join(map(str, h))
            grid.append((f"mlp_{tag}_a{alpha}", lambda h=h, alpha=alpha: make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=h, alpha=alpha, max_iter=400,
                              early_stopping=True, n_iter_no_change=15,
                              validation_fraction=0.12, random_state=seed))))
    for lr in (0.05, 0.1):
        for leaves in (31, 63):
            grid.append((f"hgb_lr{lr}_l{leaves}", lambda lr=lr, leaves=leaves:
                         HistGradientBoostingClassifier(
                             learning_rate=lr, max_leaf_nodes=leaves, max_iter=400,
                             early_stopping=True, validation_fraction=0.12,
                             random_state=seed)))
    return grid


def scores(y, p):
    return {"ap": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p))}


def bootstrap_ap(y, p, n_boot=2000, seed=SEED):
    """Percentile bootstrap CI for average precision.

    With only ~339 positives in validation, differences of a couple of AP points
    between configs are not meaningful. This makes that explicit instead of
    letting the sweep's argmax imply a precision it does not have.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:            # degenerate resample
            vals[b] = np.nan
            continue
        vals[b] = average_precision_score(y[idx], p[idx])
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="evaluate the selected model on TEST (do this once)")
    ap.add_argument("--blocks", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output csv, so that "
                    "parallel per-block runs do not overwrite each other")
    ap.add_argument("--pick", default="emb_mean:logreg_C10.0",
                    help="'block:model' to evaluate on test with --final")
    args = ap.parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)

    df, emb = load()
    blocks = feature_blocks(df, emb)
    if args.blocks:
        blocks = {k: v for k, v in blocks.items() if k in args.blocks}

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    y = df["produced"].to_numpy().astype(int)
    print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()} | "
          f"positive rate {y.mean():.4f}")

    rows = []
    for bname, X in blocks.items():
        for mname, factory in model_grid():
            t0 = time.time()
            clf = factory()
            try:
                clf.fit(X[tr], y[tr])
                pv = proba(clf, X[va])
                s = scores(y[va], pv)
            except Exception as e:                       # keep the sweep going
                print(f"  {bname:18s} {mname:20s} FAILED {type(e).__name__}: {e}")
                continue
            rows.append({"block": bname, "model": mname, "n_features": X.shape[1],
                         "val_ap": s["ap"], "val_roc_auc": s["roc_auc"],
                         "fit_s": round(time.time() - t0, 1)})
            print(f"  {bname:18s} {mname:20s} val AP {s['ap']:.4f}  "
                  f"ROC-AUC {s['roc_auc']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    res = pd.DataFrame(rows).sort_values("val_ap", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(RESULT_DIR, f"val_sweep{args.tag}.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n=== top 15 by validation AP ===")
    print(res.head(15).to_string(index=False))
    print("\n=== best per feature block ===")
    best = res.loc[res.groupby("block")["val_ap"].idxmax()].sort_values(
        "val_ap", ascending=False)
    print(best.to_string(index=False))

    if not args.final:
        print("\nTest set untouched. Re-run with --final once a model is chosen.")
        return 0

    # ---- single, final test evaluation ------------------------------------
    # Exclude val_sweep_merged.csv: it is a concatenation of the per-block files,
    # so including it would double-count every row.
    allres = pd.concat(
        [pd.read_csv(os.path.join(RESULT_DIR, f))
         for f in sorted(os.listdir(RESULT_DIR))
         if f.startswith("val_sweep_") and f.endswith(".csv") and "merged" not in f],
        ignore_index=True).sort_values("val_ap", ascending=False).reset_index(drop=True)
    print(f"\nselected from {len(allres)} validation results across all blocks")

    # The pick is DELIBERATE, not the raw argmax. analyze_val.py shows the top
    # five configs are statistically indistinguishable on validation (paired
    # bootstrap of emb_mean+modules vs emb_mean: +0.006 AP, CI [-0.039, +0.051]),
    # so the simplest and cheapest of that group is chosen. Recording it as a
    # flag keeps the decision auditable rather than silently seed-dependent.
    block, model = args.pick.split(":")
    row = allres[(allres["block"] == block) & (allres["model"] == model)]
    if row.empty:
        print(f"!!! {args.pick} not found in the validation sweep", file=sys.stderr)
        return 1
    pick = row.iloc[0]
    print(f"\n=== FINAL: {pick['block']} + {pick['model']} (val AP {pick['val_ap']:.4f}) ===")
    X = blocks[pick["block"]]
    factory = dict(model_grid())[pick["model"]]
    # Refit on train+val, which is standard once selection is frozen.
    fit = tr | va
    clf = factory()
    clf.fit(X[fit], y[fit])
    s_te = scores(y[te], proba(clf, X[te]))
    print(f"TEST  AP {s_te['ap']:.4f}  ROC-AUC {s_te['roc_auc']:.4f}")
    print(f"(baseline AP = positive rate = {y[te].mean():.4f})")
    with open(os.path.join(RESULT_DIR, "test_result.json"), "w") as fh:
        json.dump({"block": pick["block"], "model": pick["model"],
                   "val_ap": float(pick["val_ap"]), "test": s_te,
                   "test_positive_rate": float(y[te].mean())}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
