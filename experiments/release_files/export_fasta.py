#!/usr/bin/env python
"""Export construct sequences from sequences.parquet to FASTA.

    python export_fasta.py --kind protein                  # all 15,985
    python export_fasta.py --kind dna --fold GxpS --split test
    python export_fasta.py --kind both --scheme family --fold TxlS --split train

Headers are `>{id}` plus the label and module list, e.g.

    >12 produced=False n_modules=6 modules=1215,1216,105,1218,1219,1222

The id is the first whitespace-delimited token, so `rec.id` in Biopython or a
`line.split()[0]` both give you the join key back to the fold and difficulty
tables.

CAREFUL WITH LABELS IN HEADERS: `produced=` is included because it is convenient
for training on the TRAIN split. If you are exporting a TEST split to hand to a
model, pass --no-labels.

Streams in batches, so it never holds the full ~470 MB of sequence in memory.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"dna": "sequence", "protein": "protein"}


def wanted_ids(scheme, fold, split):
    if fold is None:
        return None
    f = pd.read_csv(os.path.join(HERE, "folds", f"{scheme}_folds.csv"))
    sub = f[f["family"] == fold]
    if sub.empty:
        sys.exit(f"fold `{fold}` not found in {scheme}_folds.csv. Available: "
                 f"{sorted(f['family'].unique())[:8]} ...")
    if split != "all":
        sub = sub[sub["split"] == split]
    return set(sub["id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="protein", choices=["dna", "protein", "both"])
    ap.add_argument("--scheme", default="family", choices=["family", "genus", "multifamily"])
    ap.add_argument("--fold", default=None, help="restrict to one fold (default: all constructs)")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--no-labels", action="store_true",
                    help="omit produced= from the header (use for test sets)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    keep = wanted_ids(args.scheme, args.fold, args.split)
    kinds = ["dna", "protein"] if args.kind == "both" else [args.kind]
    tag = f"{args.fold}_{args.split}" if args.fold else "all"
    os.makedirs(args.outdir, exist_ok=True)

    handles, paths, n = {}, {}, 0
    for k in kinds:
        paths[k] = os.path.join(args.outdir, f"{k}_{tag}.fasta")
        handles[k] = open(paths[k], "w")

    cols = ["id", "produced", "n_modules", "modules"] + [COL[k] for k in kinds]
    pf = pq.ParquetFile(os.path.join(HERE, "sequences.parquet"))
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        b = batch.to_pydict()
        for i, cid in enumerate(b["id"]):
            if keep is not None and cid not in keep:
                continue
            head = f">{cid}"
            if not args.no_labels:
                head += f" produced={b['produced'][i]}"
            head += f" n_modules={b['n_modules'][i]} modules={b['modules'][i]}"
            for k in kinds:
                s = b[COL[k]][i]
                handles[k].write(head + "\n")
                for j in range(0, len(s), 80):
                    handles[k].write(s[j:j + 80] + "\n")
            n += 1
    for h in handles.values():
        h.close()
    for k in kinds:
        print(f"{n} records -> {paths[k]} "
              f"({os.path.getsize(paths[k])/1e6:.1f} MB)")
    if n == 0:
        print("  [warn] nothing matched -- check --fold / --split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
