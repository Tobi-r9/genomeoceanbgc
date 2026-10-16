#!/usr/bin/env python
"""The `modules+bigram` baseline: the number to beat on these folds.

    python baseline.py --primary-only            # 12 folds, ~10 min
    python baseline.py --folds TxlS XenS         # just the two hardest
    python evaluate.py --pred baseline_predictions.csv --primary-only

Standalone: needs only numpy, pandas and scikit-learn, plus `constructs.csv` and
`folds/` from this folder. It does NOT need the sequences -- the features are
built from the module list alone.

WHAT THE MODEL IS
-----------------
Two binary feature blocks, concatenated:

  modules   one column per module id, 1 if the construct uses it.
            Order-blind: a bag of modules.
  bigrams   one column per ADJACENT module pair, 1 if the construct contains it,
            with '^' and '$' sentinels so "which module is first / last" is also
            captured.

Then a gradient-boosted tree ensemble (sklearn HistGradientBoostingClassifier,
learning_rate 0.1, max_leaf_nodes 63, max_iter 400, early stopping on a 12 %
internal split).

Fit on train+val, score test exactly once. No per-fold hyperparameter search --
these settings were chosen once, on a different (random) split, and then frozen.

Verified: with the defaults this reproduces our published per-fold test AP exactly
on all 12 primary folds (median AP 0.408, lift 3.53, ROC-AUC 0.813).

WHY BIGRAMS AND NOT JUST THE BAG
--------------------------------
A bag of modules cannot distinguish two constructs built from the same modules in
a different order, and 264 constructs (1.65 %) are exactly such permutation pairs
carrying OPPOSITE labels -- so a bag-of-modules model provably cannot fit them.
Adjacent pairs are the natural fix, and they line up with the `Transitions` sheet
of the source data (from_module -> to_module success rates).

In practice the two score about the same here (median AP 0.408 vs 0.407): the
bigrams help on some folds and the extra columns cost a little on others. The bag
is a perfectly respectable baseline too, and is one line shorter.

VOCABULARY SCOPE -- read this if your numbers differ from ours
--------------------------------------------------------------
The **bigram** vocabulary is always built on TRAIN rows only: a pair unseen in
training gets no column, so the model abstains on it rather than extrapolating.

The **module** vocabulary is controlled by --module-vocab, and the choice moves
the result by ~0.03 AP, so it is worth understanding:

  all       (default) every module in the dataset gets a column. This is what our
            published tables used, so it is the setting that reproduces them.
            It is NOT a leak: a module absent from train+val is an all-zero
            column in the fitting data, so no tree can ever split on it, and it
            is provably unused at prediction time.
  trainval  only modules occurring in train+val, i.e. exactly the columns the
            model can actually learn from. Equivalent to `all` minus the
            provably-inert columns, and verified to give bit-identical AP on all
            12 primary folds -- narrower matrix, same answer.
  train     only modules occurring in TRAIN. Stricter-looking but actually worse
            on its own terms: it throws away modules that appear in `val`, which
            IS part of the fitting data, so the model is denied usable features.
            Costs ~0.03 median AP (0.399 vs 0.408 over the 12 primary folds).

Nothing about the test labels enters the feature space under any of these.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260805


def make_model():
    """The frozen configuration. `hgb_lr0.1_l63` in our result tables."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=400,
        early_stopping=True, validation_fraction=0.12, random_state=SEED)


def featurise(mod_lists, mod_mask, big_mask, use_bigrams=True):
    """(modules | bigrams) multi-hot, vocabularies restricted to the given rows."""
    mods = [sorted(set(ms)) for ms in mod_lists]
    bigs = ([list(zip(["^"] + list(ms), list(ms) + ["$"])) for ms in mod_lists]
            if use_bigrams else [[] for _ in mod_lists])

    def build(items, keep):
        vocab = sorted({t for i, ts in enumerate(items) if keep[i] for t in ts})
        idx = {t: j for j, t in enumerate(vocab)}
        rows, cols = [], []
        for i, ts in enumerate(items):
            for t in set(ts):
                j = idx.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(items), len(vocab)))
        return X, len(vocab)

    Xm, nm = build(mods, mod_mask)
    if not use_bigrams:
        return Xm.tocsr(), nm, 0
    Xb, nb = build(bigs, big_mask)
    # Returned SPARSE on purpose. HistGradientBoostingClassifier needs dense
    # input, but densifying all 15,985 rows at once costs a few GB once the
    # bigram vocabulary is large; the caller densifies train+val and test
    # separately instead. Same numbers, roughly half the peak memory.
    return sp.hstack([Xm, Xb]).tocsr(), nm, nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="family",
                    choices=["family", "genus", "multifamily"])
    ap.add_argument("--folds", nargs="*", default=None, help="explicit fold names")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--no-bigrams", action="store_true",
                    help="plain bag-of-modules (the `modules` baseline)")
    ap.add_argument("--module-vocab", default="all", choices=["all", "trainval", "train"],
                    help="rows the module vocabulary is drawn from; see the module "
                         "docstring. Default `all` reproduces our published tables.")
    ap.add_argument("--out", default="baseline_predictions.csv")
    args = ap.parse_args()

    con = pd.read_csv(os.path.join(HERE, "constructs.csv"))
    fs = pd.read_csv(os.path.join(HERE, "folds", f"{args.scheme}_folds.csv"))
    summ = pd.read_csv(os.path.join(HERE, "folds", f"{args.scheme}_folds_summary.csv"))
    mod_lists = [[m.strip() for m in str(s).split(",")] for s in con["modules"]]
    y_all = con["produced"].to_numpy().astype(int)
    name = "modules" if args.no_bigrams else "modules+bigram"

    folds = args.folds or list(summ["family"])
    if args.primary_only and "primary" in summ:
        folds = [f for f in folds if f in set(summ.loc[summ["primary"], "family"])]
    print(f"{name}: {len(folds)} fold(s) of scheme `{args.scheme}`, "
          f"module-vocab={args.module_vocab}\n")

    out, rows = [], []
    for fold in folds:
        sub = fs[fs["family"] == fold]
        if sub.empty:
            print(f"  [skip] {fold}: not in {args.scheme}_folds.csv")
            continue
        split = con["id"].map(dict(zip(sub["id"], sub["split"])))
        if split.isna().any():
            print(f"  [skip] {fold}: fold does not cover every construct")
            continue
        tr = (split == "train").to_numpy()
        tv = tr | (split == "val").to_numpy()
        te = (split == "test").to_numpy()

        mod_mask = {"all": np.ones(len(con), bool), "trainval": tv, "train": tr}[
            args.module_vocab]
        # bigram vocabulary is always TRAIN-only
        X, nm, nb = featurise(mod_lists, mod_mask, tr,
                              use_bigrams=not args.no_bigrams)
        clf = make_model()
        clf.fit(X[tv].toarray(), y_all[tv])          # refit on train+val
        p = clf.predict_proba(X[te].toarray())[:, 1]  # scored ONCE
        yte = y_all[te]

        chance = float(yte.mean())
        r = {"fold": fold, "n_test": int(te.sum()), "positives": int(yte.sum()),
             "chance_ap": chance, "ap": float(average_precision_score(yte, p)),
             "roc_auc": float(roc_auc_score(yte, p)),
             "n_module_cols": nm, "n_bigram_cols": nb}
        r["lift"] = r["ap"] / chance
        rows.append(r)
        out.append(pd.DataFrame({"fold": fold, "id": con.loc[te, "id"].to_numpy(),
                                 "score": p, "method": name}))
        print(f"  {fold:<26s} AP {r['ap']:.3f}  chance {chance:.3f}  "
              f"lift {r['lift']:.2f}  ROC-AUC {r['roc_auc']:.3f}  "
              f"({nm}+{nb} cols, train {tv.sum()} / test {te.sum()})", flush=True)

    if not rows:
        sys.exit("no folds ran")
    pd.concat(out, ignore_index=True).to_csv(args.out, index=False)
    T = pd.DataFrame(rows)
    print(f"\nmedian over {len(T)} folds:  AP {T['ap'].median():.3f}   "
          f"lift {T['lift'].median():.2f}   ROC-AUC {T['roc_auc'].median():.3f}")
    print(f"wrote {args.out}  ->  python evaluate.py --pred {args.out}"
          f"{' --primary-only' if args.primary_only else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
