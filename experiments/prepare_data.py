#!/usr/bin/env python
"""Build the train/val/test split for the BGC 'produced' classification task.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/build_nt_dataset.py     # first: reconstruct the DNA
    python experiments/prepare_data.py

Writes
  experiments/splits/split.csv            small, versioned: id, split, label, lengths
  $GO_SCRATCH/bgc_clf/dataset_split.parquet  sequences + split assignment

Decisions (agreed with the user):
  * Exact-duplicate sequences are collapsed, and sequences carrying CONFLICTING
    labels (same input marked both produced and not) are dropped -- they are
    unlearnable and would leak across splits.
  * Stratified random 70/15/15, fixed seed, so the positive rate is identical in
    all three sets. Flagged in the report as an optimistic estimate: BGCs share
    modules by construction, so near-duplicates still straddle splits.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SEED = 20260805
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
SCRATCH = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")
IN_PARQUET = os.path.join(SCRATCH, "constructed_nt_dataset.parquet")


def main() -> int:
    os.makedirs(SPLIT_DIR, exist_ok=True)
    if not os.path.exists(IN_PARQUET):
        print(f"!!! {IN_PARQUET} missing -- run experiments/build_nt_dataset.py first",
              file=sys.stderr)
        return 1

    raw = pd.read_parquet(IN_PARQUET)
    print(f"raw constructs: {len(raw)}  positives: {raw['produced'].sum()} "
          f"({raw['produced'].mean():.4f})")

    # ---- dedup + drop label-conflicting duplicates ------------------------
    # Deduplication is on the NUCLEOTIDE sequence, i.e. exactly what the model
    # sees. (Reported for the protein too, since the two can differ: distinct
    # module combinations may encode the same peptide.)
    for col in ("sequence", "protein"):
        nuniq = raw[col].nunique()
        conf = raw.groupby(col)["produced"].nunique()
        print(f"  unique by {col:8s}: {nuniq:6d}  duplicate rows: {len(raw)-nuniq:5d}"
              f"  label-conflicting: {int((conf > 1).sum())}")

    nunique = raw.groupby("sequence")["produced"].nunique()
    conflicting = set(nunique.index[nunique > 1])
    df = (raw[~raw["sequence"].isin(conflicting)]
          .drop_duplicates(subset="sequence", keep="first")
          .reset_index(drop=True))
    df["produced"] = df["produced"].astype(bool)
    print(f"\nafter dedup + dropping {len(conflicting)} conflicting sequences: "
          f"{len(df)}  positives: {df['produced'].sum()} ({df['produced'].mean():.4f})")

    # ---- stratified split -------------------------------------------------
    # Per-class shuffle with exact per-class counts, so the positive rate is as
    # close to identical across splits as integer rounding allows.
    rng = np.random.default_rng(SEED)
    df["split"] = ""
    for label in (False, True):
        idx = np.array(df.index[df["produced"] == label], copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(FRACS["train"] * n))
        n_val = int(round(FRACS["val"] * n))
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"
    assert (df["split"] != "").all()

    print("\nsplit summary")
    s = df.groupby("split")["produced"].agg(n="size", positives="sum", positive_rate="mean")
    s["negatives"] = s["n"] - s["positives"]
    print(s.loc[["train", "val", "test"]][["n", "negatives", "positives", "positive_rate"]]
          .to_string(formatters={"positive_rate": "{:.5f}".format}))
    rates = s["positive_rate"]
    print(f"max positive-rate deviation across splits: {rates.max() - rates.min():.6f}")

    # ---- write ------------------------------------------------------------
    df[["id", "split", "produced", "n_modules", "aa_len", "nt_len"]].to_csv(
        os.path.join(SPLIT_DIR, "split.csv"), index=False)
    print(f"\nwrote {os.path.join(SPLIT_DIR, 'split.csv')}")
    out = os.path.join(SCRATCH, "dataset_split.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
