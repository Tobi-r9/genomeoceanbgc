#!/usr/bin/env python
"""Harder held-out splits: multi-family groups, and leave-one-genus-out.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/prepare_group_splits.py --mode multifamily
    python experiments/prepare_group_splits.py --mode genus

Writes, for MODE in {multifamily, genus}:
    experiments/splits/<MODE>_splits.csv          family, id, split
    experiments/splits/<MODE>_splits_summary.csv  per-fold sizes and diagnostics

Both use the same schema as family_splits.csv, so every downstream script reads
them unchanged via `--split-name`.

WHY
---
The leave-one-family-out split holds out one biosynthetic family at a time. But a
construct is a chimera of 3-4 families, so holding out one removes only about
**one of its six modules** -- the test is real but partial. These two schemes turn
that dial from different directions:

  multifamily  hold out a RANDOM group of three families at once. Random, not
               biologically related, deliberately: the genus mode below already
               covers the biological grouping, and using related families for
               both would make the two schemes redundant. This isolates the
               "more unseen chemistry per construct" effect.

  genus        hold out every family belonging to one genus. Coarser and
               biologically nested -- the natural "can it transfer to a different
               organism" question. Only 7 genera leave enough data to be usable;
               Xenorhabdus and Photorhabdus are excluded because holding either
               out would put >45 % of the dataset in test.

Everything else -- module-disjointness, hybrid labels counting for both parents,
`mix`/`unknown` never held out, validation itself being held out by group, and
the split being label-blind -- is inherited from prepare_family_splits.py and is
documented in FAMILY_SPLIT.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_family_splits import (  # noqa: E402
    DONORS, MAX_LEAK_PRIMARY, MAX_TEST_FRAC, MIN_TEST, MIN_TEST_POSITIVES,
    MIN_TEST_PRIMARY, MIN_TRAIN, NOT_FAMILIES, SEED, SPLIT_DIR,
    choose_val_families, max_jaccard, module_families, one_swap_leakage)

GROUP_SIZE = 3          # families per multifamily group


def genus_map() -> dict[str, set[str]]:
    """module number -> set of genus labels (same cleaning rules as `cluster`)."""
    don = pd.read_excel(DONORS)
    don["module number"] = don["module number"].astype(str).str.strip()
    out: dict[str, set[str]] = {}
    for mod, g in zip(don["module number"], don["genus"]):
        vals = {p.strip() for p in str(g).split("/")
                if p.strip().lower() not in NOT_FAMILIES}
        out[mod] = vals
    return out


def build_groups(mode: str, mod_sets, rng):
    """-> (group_name -> set of member labels), and per-construct set of groups."""
    mf = module_families(DONORS) if mode == "multifamily" else genus_map()
    per_construct = [set().union(*(mf.get(m, set()) for m in l)) if l else set()
                     for l in mod_sets]
    labels = sorted({x for s in per_construct for x in s})
    if mode == "genus":
        groups = {g: {g} for g in labels}
    else:
        order = list(labels)
        rng.shuffle(order)
        groups = {"+".join(sorted(order[i:i + GROUP_SIZE])): set(order[i:i + GROUP_SIZE])
                  for i in range(0, len(order), GROUP_SIZE)}
    group_sets = [{g for g, members in groups.items() if s & members}
                  for s in per_construct]
    return groups, group_sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("multifamily", "genus"), required=True)
    args = ap.parse_args()

    base = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    ids = base["id"].to_numpy()
    y = base["produced"].to_numpy().astype(bool)
    mod_lists = [[t.strip() for t in str(s).split(",")] for s in base["modules"]]
    mod_sets = [sorted(set(l)) for l in mod_lists]

    rng0 = np.random.default_rng(SEED)
    groups, group_sets = build_groups(args.mode, mod_sets, rng0)
    touch = {g: np.array([g in s for s in group_sets]) for g in groups}
    print(f"{args.mode}: {len(groups)} candidate groups over {len(base)} constructs")

    vocab = sorted({m for l in mod_sets for m in l})
    vi = {m: j for j, m in enumerate(vocab)}
    A = sp.csr_matrix(
        (np.ones(sum(map(len, mod_sets)), np.int32),
         ([i for i, l in enumerate(mod_sets) for _ in l], [vi[m] for l in mod_sets for m in l])),
        shape=(len(mod_sets), len(vocab)))
    sizes = np.asarray(A.sum(1)).ravel()

    eligible = []
    for g, mask in touch.items():
        n_test, n_pool = int(mask.sum()), int((~mask).sum())
        if (n_test >= MIN_TEST and int(y[mask].sum()) >= MIN_TEST_POSITIVES
                and n_pool >= MIN_TRAIN and n_test <= MAX_TEST_FRAC * len(base)):
            eligible.append((g, n_test))
    eligible.sort(key=lambda t: -t[1])
    print(f"{len(eligible)} groups meet the fold criteria\n")

    long_rows, summary = [], []
    for g, _ in eligible:
        te_mask = touch[g]
        pool_idx = np.where(~te_mask)[0]
        rng = np.random.default_rng(zlib.crc32(g.encode()) ^ SEED)
        val_groups = choose_val_families(pool_idx, group_sets,
                                         [x for x in groups if x != g], rng)
        va_mask = np.zeros(len(base), bool)
        for vg in val_groups:
            va_mask |= touch[vg]
        va_mask &= ~te_mask
        tr_mask = ~te_mask & ~va_mask

        tr_i, va_i, te_i = (np.where(m)[0] for m in (tr_mask, va_mask, te_mask))
        if len(va_i) == 0 or y[va_i].sum() == 0 or y[tr_i].sum() == 0:
            print(f"  [skip] {g}: degenerate validation split")
            continue
        leak = one_swap_leakage(mod_lists, tr_i, te_i)
        mj = max_jaccard(A, sizes, te_i, tr_i)
        tr_mods = {m for i in tr_i for m in mod_sets[i]}
        te_mods = {m for i in te_i for m in mod_sets[i]}

        split = np.array(["train"] * len(base), dtype=object)
        split[va_mask] = "val"; split[te_mask] = "test"
        long_rows.append(pd.DataFrame({"family": g, "id": ids, "split": split}))
        summary.append({
            "family": g, "n_members": len(groups[g]), "members": ";".join(sorted(groups[g])),
            "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
            "pos_train": round(float(y[tr_i].mean()), 4),
            "pos_val": round(float(y[va_i].mean()), 4),
            "pos_test": round(float(y[te_i].mean()), 4),
            "test_positives": int(y[te_i].sum()), "val_positives": int(y[va_i].sum()),
            "test_modules": len(te_mods),
            "test_modules_unseen_in_train": len(te_mods - tr_mods),
            "one_swap_leak_pct": round(leak, 1),
            "max_jaccard_median": round(float(np.median(mj)), 3),
            "test_frac_maxjac_ge_0.7": round(float((mj >= 0.7).mean()), 3),
            "n_val_groups": len(val_groups),
            "primary": bool(leak <= MAX_LEAK_PRIMARY and len(te_i) >= MIN_TEST_PRIMARY),
        })
        print(f"  {g:<28s} train {len(tr_i):5d} | val {len(va_i):5d} | test {len(te_i):5d} "
              f"({y[te_i].mean():.3f}) | leak {leak:5.1f}% | unseen mods "
              f"{len(te_mods - tr_mods):3d}", flush=True)

    out_long = os.path.join(SPLIT_DIR, f"{args.mode}_splits.csv")
    pd.concat(long_rows, ignore_index=True).to_csv(out_long, index=False)
    summ = pd.DataFrame(summary)
    out_sum = os.path.join(SPLIT_DIR, f"{args.mode}_splits_summary.csv")
    summ.to_csv(out_sum, index=False)
    print(f"\n{len(summ)} folds written; {int(summ['primary'].sum())} primary")
    print(f"wrote {out_long}\nwrote {out_sum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
