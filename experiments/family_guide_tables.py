#!/usr/bin/env python
"""Regenerate the result tables inside FAMILY_GUIDE.md.

    python experiments/family_guide_tables.py

Pulls every method's per-fold numbers together and rewrites the table block in
FAMILY_GUIDE.md in place. Sources:

  results/family/all_methods_ci.csv   emb_mean, symbolic, thesis, modules(+bigram)
  results/family/copytwin.csv         the memorisation floor (see below)
  results/family/<fold>_simple.csv    the minimal method
  splits/family_splits_summary.csv    twin rate per fold

`copytwin` is not a model. For each test recipe it looks for a training recipe
differing by exactly one module in one position and copies that neighbour's
label (averaging if there are several); recipes with no such neighbour get the
training base rate. It is the score obtainable by pure memorisation, so it is the
floor a real method has to clear. On a fold with no twins it lands exactly on
chance, which is the check that the split is doing its job.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM = os.path.join(REPO, "experiments", "results", "family")
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
GUIDE = os.path.join(REPO, "experiments", "FAMILY_GUIDE.md")

ORDER = ["copytwin", "emb_mean", "symbolic", "thesis", "modules+bigram", "modules", "simple"]
NICE = {"copytwin": "Copy-twin*", "emb_mean": "GenomeOcean", "symbolic": "Symbolic",
        "thesis": "Thesis", "modules+bigram": "Modules+bigram", "modules": "Modules",
        "simple": "Simple"}
TUNABLE = ["emb_mean", "modules+bigram", "modules", "simple"]

# Two independent notions of "a hard fold", used for the extra median rows.
# They are almost uncorrelated (r = 0.15 across the primary folds) and select
# overlapping but different sets, so both are shown: if a ranking survives both,
# it is not an artefact of how difficulty was defined.
CLEANEST_MAX_TWIN = 35.0     # least contaminated: hardest to score by memorising
NOVEL_MIN_HELD = 0.33        # most unseen chemistry per test construct
MIN_FOLDS_FOR_MEDIAN = 3


def difficulty():
    f = os.path.join(FAM, "fold_difficulty.csv")
    return pd.read_csv(f).set_index("family") if os.path.exists(f) else None


def extra_medians(sub, methods, value_col, pad):
    """median rows over the cleanest / most-novel folds, when enough qualify."""
    diff = difficulty()
    if diff is None:
        return []
    have = set(sub["family"])
    out = []
    for label, sel in (
        (f"median, cleanest folds (twin ≤ {CLEANEST_MAX_TWIN:.0f} %)",
         [f for f in have if f in diff.index and diff.loc[f, "twin"] <= CLEANEST_MAX_TWIN]),
        (f"median, most-novel folds (≥ {NOVEL_MIN_HELD:.0%} of construct unseen)",
         [f for f in have if f in diff.index and diff.loc[f, "frac_held"] >= NOVEL_MIN_HELD]),
    ):
        if len(sel) < MIN_FOLDS_FOR_MEDIAN:
            continue
        row = [f"**{label}**"] + [""] * pad
        for m in methods:
            g = sub[(sub["method"] == m) & (sub["family"].isin(sel))]
            row.append(f"{g[value_col].median():.3f}" if len(g) else "—")
        out.append("| " + " | ".join(row) + " |")
    return out


def load() -> pd.DataFrame:
    parts = [pd.read_csv(os.path.join(FAM, "all_methods_ci.csv"))]
    ct = os.path.join(FAM, "copytwin.csv")
    if os.path.exists(ct):
        parts.append(pd.read_csv(ct))
    simple = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FAM, "*_simple.csv")))]
    if simple:
        s = pd.concat(simple, ignore_index=True)
        meta = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))
        s = s.merge(meta[["family", "primary", "one_swap_leak_pct"]], on="family", how="left")
        s = s.rename(columns={"one_swap_leak_pct": "leak"})
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    # all_methods_ci.csv now contains copytwin and simple rows as well, so the
    # separate copytwin.csv / *_simple.csv sources would double-count them.
    # Identical duplicates would average to the same value in pivot_table and
    # hide the problem, so drop them explicitly.
    d = d.drop_duplicates(subset=["family", "arm", "method"], keep="first")
    return d[d["method"].isin(ORDER)]


def ap_block(sub, methods, arm, title):
    sub = sub[sub["arm"] == arm]
    methods = [m for m in methods if m in set(sub["method"])]
    fams = (sub[["family", "chance_ap", "n_test", "test_positives", "leak"]]
            .drop_duplicates("family").sort_values("n_test", ascending=False))
    lines = [f"**{title}**", "",
             "| Family | twin rate | test size | producers | chance AP | "
             + " | ".join(NICE[m] for m in methods) + " |",
             "|" + "|".join("---" for _ in range(5 + len(methods))) + "|"]
    for _, r in fams.iterrows():
        real = sub[(sub["family"] == r.family) & (sub["method"] != "copytwin")]
        best = real["test_ap"].max() if len(real) else None
        cells = [r.family, f"{r.leak:.0f} %", f"{int(r.n_test)}",
                 f"{int(r.test_positives)}", f"{r.chance_ap:.3f}"]
        for m in methods:
            g = sub[(sub["family"] == r.family) & (sub["method"] == m)]
            if g.empty:
                cells.append("—"); continue
            g = g.iloc[0]
            ap = f"{g.test_ap:.3f}"
            if m != "copytwin" and best is not None and g.test_ap == best:
                ap = f"**{ap}**"
            cells.append(f"{ap} [{g.ap_lo:.2f}–{g.ap_hi:.2f}]")
        lines.append("| " + " | ".join(cells) + " |")
    med = ["**median**", "", "", "", f"{fams.chance_ap.median():.3f}"]
    for m in methods:
        g = sub[sub["method"] == m]
        med.append(f"{g['test_ap'].median():.3f}" if len(g) else "—")
    lines.append("| " + " | ".join(med) + " |")
    lines += extra_medians(sub, methods, "test_ap", 4)
    return "\n".join(lines)


def auc_block(sub, methods, arm, title):
    sub = sub[sub["arm"] == arm]
    methods = [m for m in methods if m in set(sub["method"])]
    fams = (sub[["family", "n_test", "leak"]].drop_duplicates("family")
            .sort_values("n_test", ascending=False))
    lines = [f"**{title}**", "",
             "| Family | twin rate | " + " | ".join(NICE[m] for m in methods) + " |",
             "|" + "|".join("---" for _ in range(2 + len(methods))) + "|"]
    for _, r in fams.iterrows():
        real = sub[(sub["family"] == r.family) & (sub["method"] != "copytwin")]
        best = real["test_roc_auc"].max() if len(real) else None
        cells = [r.family, f"{r.leak:.0f} %"]
        for m in methods:
            g = sub[(sub["family"] == r.family) & (sub["method"] == m)]
            if g.empty:
                cells.append("—"); continue
            v = g.iloc[0].test_roc_auc
            cells.append(f"**{v:.3f}**" if (m != "copytwin" and best is not None and v == best)
                         else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    med = ["**median**", ""]
    for m in methods:
        g = sub[sub["method"] == m]
        med.append(f"{g['test_roc_auc'].median():.3f}" if len(g) else "—")
    lines.append("| " + " | ".join(med) + " |")
    lines += extra_medians(sub, methods, "test_roc_auc", 1)
    return "\n".join(lines)


DEDUP_NOTE = """### The same folds, with the near-duplicates removed

Every table above is scored on the full test set, which still contains recipes
that have a near-identical relative in training. The tables below drop those:
each fold keeps only test recipes with **no** one-module-substitution neighbour
anywhere in its training set, and everything is re-scored on what remains.

Nothing was refit — this is a filter of the evaluation set only, and it is
label-blind (having a twin depends on which modules a recipe uses, never on
whether it produced).

Two things change when you deduplicate, which is why the chance column matters
even more here. The test set shrinks, and its **positive rate moves** — on PdpS
it rises from 0.096 to 0.127, on KolS it falls from 0.155 to 0.118. So an AP here
is not comparable with the same method's AP in the tables above; compare each
number with the chance AP in its own row.

`kept` is the share of the original test set that survived. Folds marked † keep
fewer than 30 producers, which is too few to support any claim — they are shown
for completeness only."""


def dedup_block(dd, methods, kind, title):
    meta = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv")).set_index("family")
    methods = [m for m in methods if m in set(dd["method"])]
    fams = (dd[["family", "chance_ap", "n_test", "test_positives", "leak", "enough"]]
            .drop_duplicates("family").sort_values("n_test", ascending=False))
    val = "test_ap" if kind == "ap" else "test_roc_auc"
    head = (["Family", "orig twin rate", "kept", "test size", "producers", "chance AP"]
            if kind == "ap" else ["Family", "orig twin rate", "kept"])
    lines = [f"**{title}**", "",
             "| " + " | ".join(head + [NICE[m] for m in methods]) + " |",
             "|" + "|".join("---" for _ in range(len(head) + len(methods))) + "|"]
    for _, r in fams.iterrows():
        real = dd[(dd["family"] == r.family) & (dd["method"] != "copytwin")]
        best = real[val].max() if len(real) else None
        kept = 100.0 * r.n_test / meta.loc[r.family, "n_test"]
        name = r.family + ("" if r.enough else " †")
        cells = ([name, f"{r.leak:.0f} %", f"{kept:.0f} %", f"{int(r.n_test)}",
                  f"{int(r.test_positives)}", f"{r.chance_ap:.3f}"]
                 if kind == "ap" else [name, f"{r.leak:.0f} %", f"{kept:.0f} %"])
        for m in methods:
            g = dd[(dd["family"] == r.family) & (dd["method"] == m)]
            if g.empty:
                cells.append("—"); continue
            g = g.iloc[0]
            txt = f"{g[val]:.3f}"
            if m != "copytwin" and best is not None and g[val] == best:
                txt = f"**{txt}**"
            if kind == "ap":
                txt += f" [{g.ap_lo:.2f}–{g.ap_hi:.2f}]"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    med = (["**median**", "", "", "", "", f"{fams.chance_ap.median():.3f}"]
           if kind == "ap" else ["**median**", "", ""])
    for m in methods:
        g = dd[dd["method"] == m]
        med.append(f"{g[val].median():.3f}" if len(g) else "—")
    lines.append("| " + " | ".join(med) + " |")
    lines += extra_medians(dd, methods, val, 5 if kind == "ap" else 2)
    return "\n".join(lines)


def main() -> int:
    d = load()
    out = []
    for tier, lab in ((True, "PRIMARY"), (False, "SECONDARY")):
        s = d[d["primary"] == tier]
        if s.empty:
            continue
        out += [ap_block(s, ORDER, "frozen", f"{lab} — average precision (AP), default settings"), "",
                auc_block(s, ORDER, "frozen", f"{lab} — ROC-AUC, default settings"), ""]
        if tier:
            ddp = os.path.join(FAM, "dedup_test.csv")
            if os.path.exists(ddp):
                dd = pd.read_csv(ddp)
                dd = dd[dd["arm"] == "frozen"]
                out += [DEDUP_NOTE, "",
                        dedup_block(dd, ORDER, "ap",
                                    "PRIMARY, DEDUPLICATED — average precision (AP), "
                                    "default settings"), "",
                        dedup_block(dd, ORDER, "auc",
                                    "PRIMARY, DEDUPLICATED — ROC-AUC, default settings"), ""]
    for tier, lab in ((True, "PRIMARY"), (False, "SECONDARY")):
        s = d[d["primary"] == tier]
        if s.empty:
            continue
        out += [ap_block(s, TUNABLE, "tuned", f"{lab} — AP, retuned per fold"), "",
                auc_block(s, TUNABLE, "tuned", f"{lab} — ROC-AUC, retuned per fold"), ""]
    tables = "\n".join(out)

    txt = open(GUIDE).read().split("\n")
    start = next(i for i, l in enumerate(txt) if l.startswith("**PRIMARY — average precision"))
    end = next(i for i, l in enumerate(txt) if l.startswith("## 8. "))
    open(GUIDE, "w").write("\n".join(txt[:start] + [tables] + txt[end:]))
    print(f"rewrote tables in FAMILY_GUIDE.md ({d['method'].nunique()} methods, "
          f"{d['family'].nunique()} folds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
