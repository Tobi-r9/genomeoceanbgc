#!/usr/bin/env python
"""Build a self-contained, shareable copy of the NRPS evaluation splits.

    python experiments/make_release.py [--out release/nrps_splits_v1]

WHY THIS EXISTS
---------------
A collaborator wants to benchmark their own encoder on the same evaluation we
used. Handing over `experiments/splits/*.csv` alone is not enough: the fold
files carry ids but no labels, no sequences, and none of the difficulty
annotations that make the numbers interpretable. This assembles one folder that
is enough to reproduce our evaluation and add a new model to it, without any
access to this repo.

Read-only with respect to everything else. It copies and derives; it does not
modify any split, result or script.

WHAT IT DERIVES (the part that is not just a file copy)
------------------------------------------------------
Per (fold, test construct), three label-blind difficulty annotations that were
computed on the fly inside family_jaccard.py / prepare_family_splits.py and
never saved per construct:

  max_jaccard_to_train        max Jaccard of this construct's module SET against
                              every training construct in that fold. Set-based,
                              so it catches substitutions, insertions, deletions
                              and reorderings in one number.
  has_one_swap_twin_in_train  a training construct of the same length differing
                              in exactly one position. This is the positional
                              leak measure; it has a blind spot for
                              insertions/deletions, which is why max Jaccard is
                              the one to trust. See summaries/01.
  n_modules_unseen_in_train   how many of its modules never occur in training.

All three depend only on module identities, never on labels.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
RESULTS = os.path.join(REPO, "experiments", "results")
SCRATCH = os.path.join(os.environ.get("GO_SCRATCH",
                                      "/e/scratch/profound/thoeppe/genomeocean"), "bgc_clf")
SCHEMES = ("family", "genus", "multifamily")


def module_matrix(mod_sets):
    vocab = sorted({m for l in mod_sets for m in l})
    vi = {m: j for j, m in enumerate(vocab)}
    rows = [i for i, l in enumerate(mod_sets) for _ in l]
    cols = [vi[m] for l in mod_sets for m in l]
    A = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)),
                      shape=(len(mod_sets), len(vocab)))
    return A, np.asarray(A.sum(1)).ravel()


def max_jaccard(A, sizes, tr, te, chunk=500):
    """max Jaccard of every row in `te` against any row in `tr`."""
    At, st = A[tr], sizes[tr]
    best = np.zeros(len(te))
    for s0 in range(0, len(te), chunk):
        r = te[s0:s0 + chunk]
        B = (A[r] @ At.T).tocoo()
        if B.nnz:
            jac = B.data / (sizes[r][B.row] + st[B.col] - B.data)
            np.maximum.at(best[s0:s0 + chunk], B.row, jac)
    return best


def one_swap_twin(ordered, tr, te):
    """For each test index, is there a train construct differing in exactly 1 position?

    Module strings are unique across the dataset (15,985 distinct strings for
    15,985 constructs), so a wildcard-pattern collision cannot be an exact
    duplicate and therefore means exactly one position differs.
    """
    pats = set()
    for i in tr:
        l = ordered[i]
        for p in range(len(l)):
            pats.add((len(l), p, l[:p] + ("*",) + l[p + 1:]))
    out = np.zeros(len(te), bool)
    for j, i in enumerate(te):
        l = ordered[i]
        out[j] = any((len(l), p, l[:p] + ("*",) + l[p + 1:]) in pats for p in range(len(l)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    # Deliberately OUTSIDE the git repo: this folder carries a 121 MB sequence
    # parquet, which has no business in version control.
    ap.add_argument("--out", default="/e/project1/scifi/thoeppe/nrps_splits_v1")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    for sub in ("", "folds", "difficulty", "baselines"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # ---- constructs + sequences -------------------------------------------
    # Light columns only. The DNA + protein strings are ~500 MB in memory and
    # round-tripping them through pandas is what got an earlier attempt killed on
    # a login node, so the sequence file is copied as raw bytes instead.
    src = os.path.join(SCRATCH, "dataset_split.parquet")
    d = pd.read_parquet(src, columns=["id", "modules", "n_modules", "produced",
                                      "aa_len", "nt_len", "split"])
    d = d.sort_values("id").reset_index(drop=True)
    d.rename(columns={"split": "random_split"}).to_csv(
        os.path.join(out, "constructs.csv"), index=False)
    shutil.copy(src, os.path.join(out, "sequences.parquet"))
    print(f"constructs.csv      {len(d)} rows")
    print(f"sequences.parquet   copied, "
          f"{os.path.getsize(os.path.join(out, 'sequences.parquet'))/1e6:.0f} MB")

    ordered = [tuple(t.strip() for t in str(s).split(",")) for s in d["modules"]]
    mod_sets = [sorted(set(l)) for l in ordered]
    A, sizes = module_matrix(mod_sets)
    ids_all = d["id"].to_numpy()
    y_all = d["produced"].to_numpy().astype(int)
    id_pos = pd.Series(np.arange(len(d)), index=ids_all)

    # ---- folds + derived difficulty ---------------------------------------
    manifest = {"n_constructs": int(len(d)), "schemes": {}}
    for scheme in SCHEMES:
        fpath = os.path.join(SPLIT_DIR, f"{scheme}_splits.csv")
        spath = os.path.join(SPLIT_DIR, f"{scheme}_splits_summary.csv")
        if not os.path.exists(fpath):
            print(f"  [skip] {scheme}: no split file")
            continue
        shutil.copy(fpath, os.path.join(out, "folds", f"{scheme}_folds.csv"))
        shutil.copy(spath, os.path.join(out, "folds", f"{scheme}_folds_summary.csv"))
        fs = pd.read_csv(fpath)
        summ = pd.read_csv(spath)

        rows = []
        for fam in summ["family"]:
            fold = fs[fs["family"] == fam]
            pos = id_pos.reindex(fold["id"]).to_numpy()
            s = fold["split"].to_numpy()
            tr = pos[s == "train"]
            te = pos[s == "test"]
            seen = np.asarray(A[tr].sum(0)).ravel() > 0
            unseen_cols = np.where(~seen)[0]
            # entries are 1 per distinct module, so a row sum over the unseen
            # columns IS the count of unseen modules
            n_unseen = (np.asarray(A[te][:, unseen_cols].sum(1)).ravel().astype(int)
                        if len(unseen_cols) else np.zeros(len(te), int))
            rows.append(pd.DataFrame({
                "fold": fam,
                "id": ids_all[te],
                "produced": y_all[te],
                "n_modules": [len(ordered[i]) for i in te],
                "max_jaccard_to_train": np.round(max_jaccard(A, sizes, tr, te), 6),
                "has_one_swap_twin_in_train": one_swap_twin(ordered, tr, te),
                "n_modules_unseen_in_train": n_unseen,
            }))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(os.path.join(out, "difficulty", f"{scheme}_test_difficulty.csv"),
                  index=False)
        manifest["schemes"][scheme] = {
            "n_folds": int(summ["family"].nunique()),
            "n_primary_folds": int(summ["primary"].sum()) if "primary" in summ else None,
            "n_test_rows_total": int(len(df)),
            "median_max_jaccard": float(df["max_jaccard_to_train"].median()),
        }
        print(f"{scheme:12s} {summ['family'].nunique():3d} folds, {len(df):6d} test rows, "
              f"median max-J {df['max_jaccard_to_train'].median():.3f}")

    # ---- our reference numbers, for comparison ----------------------------
    ref = pd.read_csv(os.path.join(RESULTS, "family", "summary_test.csv"))
    keep = ["family", "arm", "method", "test_ap", "ap_lo", "ap_hi", "test_roc_auc",
            "chance_ap", "n_test", "test_positives", "lift", "primary"]
    ref[keep].rename(columns={"family": "fold"}).to_csv(
        os.path.join(out, "baselines", "family_reference_ap.csv"), index=False)
    print(f"reference AP table  {len(ref)} (fold, arm, method) rows")

    # Two more aggregate reference tables. copytwin is the leakage baseline
    # ("copy the label of the most similar training construct") and exists only
    # per fold -- it has no per-construct score file, so it is absent from
    # family_test_scores.csv.gz.
    for src_name, dst_name in (("copytwin.csv", "family_copytwin_ap.csv"),
                               ("jaccard_test.csv", "family_jaccard_capped_ap.csv"),
                               ("dedup_test.csv", "family_dedup_ap.csv")):
        q = os.path.join(RESULTS, "family", src_name)
        if os.path.exists(q):
            shutil.copy(q, os.path.join(out, "baselines", dst_name))
            print(f"reference table     {dst_name}")

    # per-construct baseline scores, so paired comparisons are possible
    sc = []
    for fam in pd.read_csv(os.path.join(SPLIT_DIR, "family_splits_summary.csv"))["family"]:
        p = os.path.join(RESULTS, "family", f"{fam}_scores.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for k in z.files:
            if "__" not in k:
                continue
            arm, meth = k.split("__")
            sc.append(pd.DataFrame({"fold": fam, "id": z["ids"], "y": z["y"],
                                    "arm": arm, "method": meth, "score": z[k]}))
        for meth, fn, sk, ik in (("symbolic", f"{fam}_symbolic.npz", "test_scores", "test_ids"),
                                 ("thesis", f"{fam}_thesis.npz", "test_scores", "test_ids")):
            q = os.path.join(RESULTS, "family", fn)
            if os.path.exists(q):
                z = np.load(q)
                sc.append(pd.DataFrame({"fold": fam, "id": z[ik], "y": z["test_y"],
                                        "arm": "frozen", "method": meth, "score": z[sk]}))
        # `simple` carries both arms; `ensemble` stores its rank-mixed score
        # directly, so nothing is recomputed here.
        q = os.path.join(RESULTS, "family", f"{fam}_simple.npz")
        if os.path.exists(q):
            z = np.load(q)
            for arm in ("tuned", "frozen"):
                if arm in z.files:
                    sc.append(pd.DataFrame({"fold": fam, "id": z["ids"], "y": z["y"],
                                            "arm": arm, "method": "simple",
                                            "score": z[arm]}))
        q = os.path.join(RESULTS, "family", f"{fam}_ensemble.npz")
        if os.path.exists(q):
            z = np.load(q)
            sc.append(pd.DataFrame({"fold": fam, "id": z["ids"], "y": z["y"],
                                    "arm": "frozen", "method": "ensemble",
                                    "score": z["ensemble"]}))
    if sc:
        S = pd.concat(sc, ignore_index=True)
        f = os.path.join(out, "baselines", "family_test_scores.csv.gz")
        S.to_csv(f, index=False, compression="gzip")
        print(f"baseline scores     {len(S)} rows, {os.path.getsize(f)/1e6:.1f} MB")

    # ---- the scripts and README that ship with the folder -----------------
    # Kept in the repo (experiments/release_files/) rather than written inline,
    # so this folder can be rebuilt from scratch anywhere and stays in step with
    # whatever those files say.
    rf = os.path.join(REPO, "experiments", "release_files")
    for fn in sorted(os.listdir(rf)):
        if not fn.startswith("."):
            shutil.copy(os.path.join(rf, fn), os.path.join(out, fn))
            print(f"shipped              {fn}")

    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
