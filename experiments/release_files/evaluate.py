#!/usr/bin/env python
"""Score a model on the NRPS leave-one-family-out folds.

    python evaluate.py --pred my_scores.csv
    python evaluate.py --pred my_scores.csv --scheme genus --name my_encoder

Your --pred file needs three columns, one row per (fold, test construct):

    fold,id,score
    GxpS,12,0.83
    GxpS,17,0.11
    ...

`score` is any real number where higher = more likely to produce. It does NOT
need to be a calibrated probability -- every metric here is rank-based.

Run one model at a time per file, or add a `method` column and this will report
each method separately.

This script depends only on numpy, pandas and scikit-learn. It needs no access
to the repository that produced the splits.

WHAT IT REPORTS, AND WHY
------------------------
Average precision (AP) is the headline number, but AP is NOT comparable across
folds: chance-level AP equals the fold's positive rate, which ranges from 0.036
to 0.226 here. So everything is also reported as **lift = AP / chance_AP**, and
cross-fold aggregates use lift or a per-fold median, never a pooled AP.

Each fold is also scored on Jaccard-capped subsets (J <= 0.7 / 0.6 / 0.5), which
drop test constructs that closely resemble something in that fold's training
set. A model that leans on near-duplicates loses ground as the cap tightens.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 2000
CAPS = (0.7, 0.6, 0.5)
MIN_N, MIN_POS = 20, 5          # below this a fold subset is not scored


def boot_ci(y, s, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap CI for AP. Resamples rows, not scores."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        out.append(average_precision_score(y[i], s[i]))
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired(y, a, b, n_boot=N_BOOT, seed=0):
    """AP(a) - AP(b) with a CI, both scored on the SAME resampled rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        d.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    if not d:
        return np.nan, np.nan, np.nan
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def prec_at_k(y, s, k=50):
    if len(y) < k:
        return np.nan
    return float(y[np.argsort(-s)[:k]].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV with fold,id,score[,method]")
    ap.add_argument("--scheme", default="family", choices=["family", "genus", "multifamily"])
    ap.add_argument("--name", default="my_model", help="label used when there is no method column")
    ap.add_argument("--out", default=None, help="write the per-fold table here")
    ap.add_argument("--primary-only", action="store_true",
                    help="restrict to the folds flagged primary (large enough to trust)")
    ap.add_argument("--compare", default="modules+bigram",
                    help="baseline method to paired-test against, or 'none'")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    diff = pd.read_csv(os.path.join(HERE, "difficulty", f"{args.scheme}_test_difficulty.csv"))
    summ = pd.read_csv(os.path.join(HERE, "folds", f"{args.scheme}_folds_summary.csv"))
    summ = summ.rename(columns={"family": "fold"})
    if args.primary_only and "primary" in summ:
        diff = diff[diff["fold"].isin(summ.loc[summ["primary"], "fold"])]

    pred = pd.read_csv(args.pred)
    missing = {"fold", "id", "score"} - set(pred.columns)
    if missing:
        sys.exit(f"--pred is missing column(s): {sorted(missing)}")
    if "method" not in pred.columns:
        pred["method"] = args.name

    # Drop folds we are not scoring (e.g. the non-primary ones under
    # --primary-only) up front, so they do not each produce a "skipped" line.
    keep_folds = set(diff["fold"])
    off = pred[~pred["fold"].isin(keep_folds)]
    if len(off):
        print(f"  [note] ignoring {len(off)} rows in {off['fold'].nunique()} fold(s) "
              f"not being scored here: {sorted(off['fold'].unique())[:6]}"
              f"{' ...' if off['fold'].nunique() > 6 else ''}\n")
        pred = pred[pred["fold"].isin(keep_folds)]
    if pred.empty:
        sys.exit("no rows in --pred match the folds being scored")

    # ---- sanity checks on the submission -------------------------------------
    truth = diff.set_index(["fold", "id"])["produced"]
    for meth, g in pred.groupby("method"):
        key = pd.MultiIndex.from_frame(g[["fold", "id"]])
        extra = key.difference(truth.index)
        if len(extra):
            print(f"  [warn] {meth}: {len(extra)} (fold, id) rows are NOT test rows of "
                  f"the {args.scheme} split and will be ignored. First few: "
                  f"{list(extra[:3])}")
        for fold, gg in g.groupby("fold"):
            want = set(diff.loc[diff["fold"] == fold, "id"])
            got = set(gg["id"])
            if want - got:
                print(f"  [warn] {meth}/{fold}: {len(want - got)} of {len(want)} test "
                      f"constructs have no score; fold scored on the rest only.")
            if gg["id"].duplicated().any():
                sys.exit(f"{meth}/{fold}: duplicate ids in --pred")

    base = None
    bp = os.path.join(HERE, "baselines", f"{args.scheme}_test_scores.csv.gz")
    if args.compare != "none" and os.path.exists(bp):
        b = pd.read_csv(bp)
        b = b[b["method"] == args.compare]
        if len(b):
            # prefer the tuned arm where a method has one
            arm = "tuned" if (b["arm"] == "tuned").any() else b["arm"].iloc[0]
            base = b[b["arm"] == arm].set_index(["fold", "id"])["score"]
            print(f"paired comparison against baseline `{args.compare}` ({arm} arm)\n")
        else:
            print(f"  [warn] baseline `{args.compare}` not in {bp}; skipping comparison\n")
    elif args.compare != "none":
        print(f"  [note] no per-construct baseline scores shipped for scheme "
              f"`{args.scheme}` (only `family` has them), so no paired comparison. "
              f"Absolute AP / lift are still reported.\n")

    rows = []
    for (meth, fold), g in pred.groupby(["method", "fold"]):
        d = diff[diff["fold"] == fold].merge(g[["id", "score"]], on="id", how="inner")
        if len(d) < MIN_N or d["produced"].sum() < MIN_POS:
            print(f"{meth:<18s} {fold:<26s} skipped (n={len(d)}, "
                  f"pos={int(d['produced'].sum())})")
            continue
        y = d["produced"].to_numpy().astype(int)
        s = d["score"].to_numpy(float)
        lo, hi = boot_ci(y, s, args.n_boot)
        r = {"method": meth, "fold": fold, "n_test": len(y), "positives": int(y.sum()),
             "chance_ap": float(y.mean()),
             "ap": float(average_precision_score(y, s)), "ap_lo": lo, "ap_hi": hi,
             "roc_auc": float(roc_auc_score(y, s)),
             "prec_at_50": prec_at_k(y, s, 50),
             "median_max_jaccard": float(d["max_jaccard_to_train"].median()),
             "twin_pct": float(100 * d["has_one_swap_twin_in_train"].mean())}
        r["lift"] = r["ap"] / r["chance_ap"]
        for c in CAPS:
            m = (d["max_jaccard_to_train"] <= c).to_numpy()
            r[f"ap_J<={c}"] = (float(average_precision_score(y[m], s[m]))
                               if m.sum() >= MIN_N and y[m].sum() >= MIN_POS else np.nan)
        if base is not None:
            bs = base.reindex(pd.MultiIndex.from_frame(d[["fold", "id"]])).to_numpy()
            if not np.isnan(bs).any():
                m, l, h = paired(y, s, bs, args.n_boot)
                r.update({"d_ap_vs_baseline": m, "d_lo": l, "d_hi": h,
                          "beats_baseline": bool(l > 0)})
        rows.append(r)
        print(f"{meth:<18s} {fold:<26s} AP {r['ap']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"chance {r['chance_ap']:.3f}  lift {r['lift']:.2f}  "
              f"ROC-AUC {r['roc_auc']:.3f}  P@50 {r['prec_at_50']:.2f}")

    if not rows:
        sys.exit("nothing scored -- check that --pred uses the fold names and ids "
                 "from difficulty/*.csv")
    T = pd.DataFrame(rows)
    out = args.out or f"evaluation_{args.scheme}.csv"
    T.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print(f"ACROSS FOLDS  ({T['fold'].nunique()} folds"
          f"{', primary only' if args.primary_only else ''})")
    print("=" * 78)
    agg = T.groupby("method").agg(
        folds=("fold", "size"), median_ap=("ap", "median"),
        median_lift=("lift", "median"), median_roc_auc=("roc_auc", "median"),
        median_p50=("prec_at_50", "median"))
    print(agg.round(3).to_string())
    if "d_ap_vs_baseline" in T:
        print(f"\npaired vs `{args.compare}`:")
        g = T.groupby("method").agg(
            median_dAP=("d_ap_vs_baseline", "median"),
            folds_ahead=("d_ap_vs_baseline", lambda s: int((s > 0).sum())),
            folds_significant=("beats_baseline", "sum"))
        print(g.round(4).to_string())
        print("  `folds_significant` = folds where the paired 95 % CI excludes zero.")
    caps = [f"ap_J<={c}" for c in CAPS if f"ap_J<={c}" in T]
    if caps:
        print("\nmedian AP as similarity to training is capped "
              "(a model relying on near-duplicates fades here):")
        print(T.groupby("method")[["ap"] + caps].median().round(3).to_string())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
