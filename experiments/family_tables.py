#!/usr/bin/env python
"""Aggregate the per-fold family results into the tables FAMILY_RESULTS.md uses.

    python experiments/family_tables.py

Reads results/family/<fold>_test.csv, <fold>_paired.csv and <fold>_symbolic.json,
joins the fold metadata from splits/family_splits_summary.csv, and writes

  results/family/summary_test.csv    one row per (fold, arm, method)
  results/family/summary_paired.csv  one row per (fold, arm, baseline)

then prints the markdown tables.

A note on comparing folds
-------------------------
Average precision is NOT comparable across folds: each fold has its own positive
rate (0.036 to 0.226), and AP at chance equals that rate. Every table therefore
carries **lift = AP / chance_AP** alongside raw AP, and cross-fold aggregates use
lift. Raw AP is still shown because it is what REPORT.md quotes.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM_DIR = os.path.join(REPO, "experiments", "results", "family")
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
METHOD_ORDER = ["emb_mean", "symbolic", "thesis", "modules+bigram", "modules"]


def fold_files(pattern: str) -> list[str]:
    """Per-fold inputs matching `pattern`, and ONLY those.

    Aggregate outputs live in the same directory and match the same globs:
    summary_test.csv, summary_paired.csv and dedup_test.csv all end in
    `_test.csv` or `_paired.csv`. Ingesting one of those double-counts every
    fold and breaks the metadata merge (`primary` becomes `primary_x`).

    Matching on a prefix blacklist proved fragile -- dedup_test.csv slipped past
    an earlier `startswith("summary")` filter. So this allow-lists the actual
    fold names from the split summary instead, which cannot drift.
    """
    known = set(pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))["family"])
    suffix = pattern.lstrip("*")
    out = []
    for f in glob.glob(os.path.join(FAM_DIR, pattern)):
        base = os.path.basename(f)
        if base[: -len(suffix)] in known:
            out.append(f)
    return sorted(out)


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = [pd.read_csv(f) for f in fold_files("*_test.csv")]
    if not tests:
        raise SystemExit(f"no *_test.csv in {FAM_DIR} -- run family_eval.py first")
    t = pd.concat(tests, ignore_index=True)

    # symbolic arrives as one json per fold; it has no tuned arm (frozen config only)
    sym = []
    for f in fold_files("*_symbolic.json"):
        j = json.load(open(f))
        sym.append({"family": j["family"], "arm": "frozen", "method": "symbolic",
                    # read from the json, not hardcoded: 3 folds could not supply
                    # 150000 motif bins and were capped (see family_symbolic.py)
                    "model": (f"n_kmers{j['config']['n_kmers']}"
                              f"_l2{j['config']['l2']}_it{j['config']['max_iter']}"),
                    "n_kmers": j["config"]["n_kmers"],
                    "n_kmers_capped": bool(j.get("n_kmers_capped", False)),
                    "val_ap": j["val"]["ap"], "test_ap": j["test"]["ap"],
                    "ap_lo": np.nan, "ap_hi": np.nan,
                    "test_roc_auc": j["test"]["roc_auc"],
                    "chance_ap": j["test"]["chance_ap"]})
    # the thesis method (NRPS-Models-and-data) -- also frozen-arm only, since
    # their code has no hyperparameter sweep
    for f in fold_files("*_thesis.json"):
        j = json.load(open(f))
        sym.append({"family": j["family"], "arm": "frozen", "method": "thesis",
                    "model": "xgb_d6_eta0.05_r300",
                    "val_ap": j["val"]["ap"], "test_ap": j["test"]["ap"],
                    "ap_lo": np.nan, "ap_hi": np.nan,
                    "test_roc_auc": j["test"]["roc_auc"],
                    "chance_ap": j["test"]["chance_ap"]})

    if sym:
        t = pd.concat([t, pd.DataFrame(sym)], ignore_index=True)

    meta = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))
    t = t.merge(meta[["family", "n_test", "test_positives", "one_swap_leak_pct",
                      "primary"]], on="family", how="left", suffixes=("", "_meta"))
    for c in ("n_test", "test_positives"):
        if f"{c}_meta" in t:
            t[c] = t[c].fillna(t[f"{c}_meta"])
            t = t.drop(columns=[f"{c}_meta"])
    t["lift"] = t["test_ap"] / t["chance_ap"]

    pairs = [pd.read_csv(f) for f in fold_files("*_paired.csv")]
    p = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    if not p.empty:
        p = p.merge(meta[["family", "primary", "one_swap_leak_pct"]], on="family", how="left")
    return t, p


def md(df: pd.DataFrame, cols, floatfmt="{:.3f}") -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating))
                         and not np.isnan(v) else ("" if (isinstance(v, float) and np.isnan(v))
                                                   else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    t, p = collect()
    t.to_csv(os.path.join(FAM_DIR, "summary_test.csv"), index=False)
    if not p.empty:
        p.to_csv(os.path.join(FAM_DIR, "summary_paired.csv"), index=False)
    n_folds = t["family"].nunique()
    have_sym = (t["method"] == "symbolic").sum()
    have_th = (t["method"] == "thesis").sum()
    print(f"{n_folds} folds, {len(t)} (fold, arm, method) rows, "
          f"{have_sym} with symbolic, {have_th} with thesis\n")

    for tier, label in ((True, "PRIMARY"), (False, "SECONDARY")):
        sub = t[(t["primary"] == tier)]
        if sub.empty:
            continue
        print(f"\n########## {label} folds ##########")
        for arm in ("tuned", "frozen"):
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            print(f"\n--- {arm} arm: test AP by fold ---")
            piv = a.pivot_table(index="family", columns="method", values="test_ap")
            piv = piv.reindex(columns=[m for m in METHOD_ORDER if m in piv.columns])
            piv["chance"] = a.groupby("family")["chance_ap"].first()
            piv["leak%"] = a.groupby("family")["one_swap_leak_pct"].first()
            print(piv.round(3).to_string())

            print(f"\n--- {arm} arm: lift over chance (AP / positive rate) ---")
            pl = a.pivot_table(index="family", columns="method", values="lift")
            pl = pl.reindex(columns=[m for m in METHOD_ORDER if m in pl.columns])
            print(pl.round(2).to_string())
            print(f"\n  median lift: "
                  + "  ".join(f"{m} {pl[m].median():.2f}" for m in pl.columns))

    if not p.empty:
        print("\n\n########## paired bootstrap vs emb_mean (test) ##########")
        for tier, label in ((True, "PRIMARY"), (False, "SECONDARY")):
            for arm in ("tuned", "frozen"):
                a = p[(p["primary"] == tier) & (p["arm"] == arm)]
                if a.empty:
                    continue
                print(f"\n--- {label} / {arm} ---")
                g = a.groupby("vs").agg(
                    folds=("family", "size"),
                    median_dAP=("d_ap", "median"),
                    emb_wins=("d_ap", lambda s: int((s > 0).sum())),
                    significant=("significant", "sum"))
                print(g.round(4).to_string())
    print(f"\nwrote {os.path.join(FAM_DIR, 'summary_test.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
