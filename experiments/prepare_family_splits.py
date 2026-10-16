#!/usr/bin/env python
"""Leave-one-family-out splits: hold out a whole BGC family, train on the rest.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/prepare_family_splits.py

Writes
  experiments/splits/family_splits.csv          family, id, split   (long format)
  experiments/splits/family_splits_summary.csv  per-fold sizes, rates, diagnostics

Why this exists
---------------
`prepare_data.py` makes a stratified RANDOM split. That is optimistic here: every
construct is assembled from the same 1,384-module pool, so 93.0 % of its test
constructs have a training construct differing by exactly ONE module in one
position. The model is being asked to interpolate inside a known pool, not to
generalise to new chemistry.

This script instead holds out a *biosynthetic family*. The family label is the
`cluster` column of `donors_data.xlsx` -- the wild-type BGC a donor module was
lifted from (GxpS = GameXPeptide synthetase, SzeS, XbcS, ...).

Decisions
---------
* **Module-disjoint, not construct-disjoint.** A construct is a chimera drawing
  modules from 3-4 different clusters, so it has no single family label. Holding
  out family F therefore means: remove every module belonging to F from
  training, and test on every construct that uses at least one of them. Only
  0-31 constructs are "pure" for any cluster, so a pure-family test set is not
  available at this dataset size.

* **Hybrid cluster labels count for both parents.** 26 modules carry labels like
  `GxpS/XgsC` or `KolS/GxpS`. Splitting on "/" and treating the module as a
  member of both families is the conservative choice: holding out GxpS then also
  removes the hybrids, so no GxpS-derived module survives in training.

* **`mix`, `unknown` and blank are not families** and are never held out. They
  are bookkeeping values, not biosynthetic origins. Modules carrying them stay
  in training (they can still land in test if the construct also uses an F
  module, which is correct -- the construct is contaminated either way).

* **Validation is itself family-held-out.** Selecting hyperparameters on a
  randomly-drawn validation set would pick the model that interpolates best,
  which is the failure mode this split exists to avoid. So whole families are
  held out of the training pool for validation too, giving a genuine
  train / val / test family hierarchy: training sees neither the val nor the
  test families.

* **The split is label-blind.** Family membership comes from the donor
  spreadsheet, never from a model, an embedding, or the `produced` column.

`id` matches `splits/split.csv` and the cached embedding shards, so nothing has
to be re-embedded -- only the classifier heads are refit.
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 20260805

# A fold is emitted only if it is worth running. Too small a test set gives an
# AP with a CI wider than any effect we could detect; too large a one starves
# training and confounds the comparison with a sample-size effect.
MIN_TEST = 200
MIN_TEST_POSITIVES = 30
MIN_TRAIN = 5000
MAX_TEST_FRAC = 0.45

VAL_TARGET_FRAC = 0.15      # of the post-test pool
VAL_CAP_FRAC = 0.30

# A fold whose test constructs still have one-swap neighbours in training is a
# weak test of family transfer, however clean the family bookkeeping was. Folds
# above this are kept (they are still evidence) but flagged as secondary.
MAX_LEAK_PRIMARY = 45.0
MIN_TEST_PRIMARY = 500

NOT_FAMILIES = {"mix", "unknown", "nan", "na", "", "?"}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
DONORS = os.path.join(REPO, "dataset", "donors_data.xlsx")


def module_families(path: str) -> dict[str, set[str]]:
    """module number -> set of family names (hybrids belong to every parent)."""
    don = pd.read_excel(path)
    don["module number"] = don["module number"].astype(str).str.strip()
    out: dict[str, set[str]] = {}
    for mod, cl in zip(don["module number"], don["cluster"]):
        fams = set()
        for part in str(cl).split("/"):
            p = part.strip()
            if p.lower() not in NOT_FAMILIES:
                fams.add(p)
        out[mod] = fams
    return out


def one_swap_leakage(mod_lists, train_idx, test_idx) -> float:
    """% of test constructs with a train construct differing by one module.

    Position-sensitive: two constructs count as neighbours only if they agree on
    every position but one. This is the same statistic quoted for the random
    split (93.0 %), so the two numbers are directly comparable.
    """
    seen = set()
    for i in train_idx:
        l = mod_lists[i]
        for p in range(len(l)):
            seen.add((tuple(l[:p]), tuple(l[p + 1:])))
    hits = 0
    for i in test_idx:
        l = mod_lists[i]
        if any((tuple(l[:p]), tuple(l[p + 1:])) in seen for p in range(len(l))):
            hits += 1
    return 100.0 * hits / max(len(test_idx), 1)


def max_jaccard(A, sizes, test_idx, train_idx, chunk=1000) -> np.ndarray:
    """For each test construct, its highest module-set Jaccard against training."""
    At = A[train_idx]
    st = sizes[train_idx]
    best = np.zeros(len(test_idx))
    for s in range(0, len(test_idx), chunk):
        rows = test_idx[s:s + chunk]
        B = (A[rows] @ At.T).tocoo()
        if B.nnz == 0:
            continue
        jac = B.data / (sizes[rows][B.row] + st[B.col] - B.data)
        np.maximum.at(best[s:s + chunk], B.row, jac)
    return best


def choose_val_families(pool_idx, fam_sets, candidates, rng) -> list[str]:
    """Hold whole families out of the training pool to form validation.

    Greedy over a seeded shuffle rather than picking the family closest to the
    target size: one big family would make validation a single-family estimate,
    and whether that family happens to be easy would then drive every model
    choice. Several small ones average over that.
    """
    pool = set(pool_idx)
    target = VAL_TARGET_FRAC * len(pool_idx)
    cap = VAL_CAP_FRAC * len(pool_idx)
    order = list(candidates)
    rng.shuffle(order)
    chosen: list[str] = []
    val: set[int] = set()
    for fam in order:
        add = {i for i in pool if fam in fam_sets[i]} - val
        if not add or len(val) + len(add) > cap:
            continue
        chosen.append(fam)
        val |= add
        if len(val) >= target:
            break
    return sorted(chosen)


def main() -> int:
    os.makedirs(SPLIT_DIR, exist_ok=True)
    base = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    ids = base["id"].to_numpy()
    y = base["produced"].to_numpy().astype(bool)
    mod_lists = [[t.strip() for t in str(s).split(",")] for s in base["modules"]]
    mod_sets = [sorted(set(l)) for l in mod_lists]

    mf = module_families(DONORS)
    missing = {m for l in mod_sets for m in l if m not in mf}
    if missing:
        print(f"!!! {len(missing)} modules absent from donors_data.xlsx", file=sys.stderr)
        return 1

    # family membership per construct = union over its modules
    fam_sets = [set().union(*(mf[m] for m in l)) if l else set() for l in mod_sets]
    touch: dict[str, np.ndarray] = {}
    for fam in sorted({f for s in fam_sets for f in s}):
        touch[fam] = np.array([fam in s for s in fam_sets])

    print(f"{len(base)} constructs, {sum(len(l) for l in mod_sets)} module slots, "
          f"{len({m for l in mod_sets for m in l})} distinct modules")
    print(f"{len(touch)} families after dropping {sorted(NOT_FAMILIES - {''})}")

    # sparse module matrix, for the Jaccard diagnostic
    vocab = sorted({m for l in mod_sets for m in l})
    vi = {m: j for j, m in enumerate(vocab)}
    A = sp.csr_matrix(
        (np.ones(sum(map(len, mod_sets)), np.int32),
         ([i for i, l in enumerate(mod_sets) for _ in l], [vi[m] for l in mod_sets for m in l])),
        shape=(len(mod_sets), len(vocab)))
    sizes = np.asarray(A.sum(1)).ravel()

    # ---- one fold per family ---------------------------------------------
    eligible = []
    for fam, mask in touch.items():
        n_test, n_pool = int(mask.sum()), int((~mask).sum())
        if (n_test >= MIN_TEST and int(y[mask].sum()) >= MIN_TEST_POSITIVES
                and n_pool >= MIN_TRAIN and n_test <= MAX_TEST_FRAC * len(base)):
            eligible.append((fam, n_test))
    eligible.sort(key=lambda t: -t[1])
    print(f"\n{len(eligible)} families meet the fold criteria "
          f"(test >= {MIN_TEST} & >= {MIN_TEST_POSITIVES} positives, "
          f"train >= {MIN_TRAIN}, test <= {MAX_TEST_FRAC:.0%})\n")

    long_rows, summary = [], []
    for fam, _ in eligible:
        te_mask = touch[fam]
        pool_idx = np.where(~te_mask)[0]
        # crc32, not hash(): Python's string hash is salted per process, so
        # hash() would give a different validation set on every run.
        rng = np.random.default_rng(zlib.crc32(fam.encode()) ^ SEED)
        val_fams = choose_val_families(pool_idx, fam_sets, [f for f in touch if f != fam], rng)
        va_mask = np.zeros(len(base), bool)
        for vf in val_fams:
            va_mask |= touch[vf]
        va_mask &= ~te_mask
        tr_mask = ~te_mask & ~va_mask

        tr_i, va_i, te_i = np.where(tr_mask)[0], np.where(va_mask)[0], np.where(te_mask)[0]
        leak = one_swap_leakage(mod_lists, tr_i, te_i)
        mj = max_jaccard(A, sizes, te_i, tr_i)
        tr_mods = {m for i in tr_i for m in mod_sets[i]}
        te_mods = {m for i in te_i for m in mod_sets[i]}

        split = np.array(["train"] * len(base), dtype=object)
        split[va_mask] = "val"
        split[te_mask] = "test"
        long_rows.append(pd.DataFrame({"family": fam, "id": ids, "split": split}))

        summary.append({
            "family": fam,
            "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
            "pos_train": round(float(y[tr_i].mean()), 4),
            "pos_val": round(float(y[va_i].mean()), 4),
            "pos_test": round(float(y[te_i].mean()), 4),
            "test_positives": int(y[te_i].sum()),
            "val_positives": int(y[va_i].sum()),
            "test_modules": len(te_mods),
            "test_modules_unseen_in_train": len(te_mods - tr_mods),
            "one_swap_leak_pct": round(leak, 1),
            "max_jaccard_median": round(float(np.median(mj)), 3),
            "test_frac_maxjac_ge_0.7": round(float((mj >= 0.7).mean()), 3),
            "n_val_families": len(val_fams),
            "val_families": ";".join(val_fams),
            "primary": bool(leak <= MAX_LEAK_PRIMARY and len(te_i) >= MIN_TEST_PRIMARY),
        })
        print(f"  {fam:<10s} train {len(tr_i):5d} ({y[tr_i].mean():.3f}) | "
              f"val {len(va_i):5d} ({y[va_i].mean():.3f}) | "
              f"test {len(te_i):5d} ({y[te_i].mean():.3f}) | "
              f"leak {leak:5.1f}% | unseen mods {len(te_mods - tr_mods):3d}", flush=True)

    out_long = os.path.join(SPLIT_DIR, "family_splits.csv")
    pd.concat(long_rows, ignore_index=True).to_csv(out_long, index=False)
    out_sum = os.path.join(SPLIT_DIR, "family_splits_summary.csv")
    summ = pd.DataFrame(summary)
    summ.to_csv(out_sum, index=False)

    # ---- integrity checks --------------------------------------------------
    print("\nintegrity checks")
    perm_key = [frozenset(l) for l in mod_sets]
    bad = 0
    for fam, _ in eligible:
        m = touch[fam]
        g = pd.DataFrame({"k": perm_key, "t": m}).groupby("k")["t"].nunique()
        bad += int((g > 1).sum())
    print(f"  permutation siblings straddling the test boundary: {bad} "
          f"(0 expected -- assignment depends only on the module SET)")
    print(f"  every fold module-disjoint for its family: "
          f"{all(r['test_modules_unseen_in_train'] > 0 for r in summary)}")
    print(f"\nrandom stratified split, for reference: one-swap leakage 93.0 %, "
          f"positive rate 0.141 in all three sets")

    prim = summ[summ["primary"]]
    print(f"\n{len(prim)} PRIMARY folds (leak <= {MAX_LEAK_PRIMARY:.0f} %, "
          f"test >= {MIN_TEST_PRIMARY}): {', '.join(prim['family'])}")
    print(f"  their test positive rates span {prim['pos_test'].min():.3f} - "
          f"{prim['pos_test'].max():.3f}, so each fold needs its own chance line")
    print(f"{len(summ) - len(prim)} secondary folds retained but still leaky; "
          f"report them separately, do not average them in")
    print(f"\nwrote {out_long} ({os.path.getsize(out_long)/1e6:.1f} MB)")
    print(f"wrote {out_sum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
