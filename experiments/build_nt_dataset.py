#!/usr/bin/env python
"""Reconstruct NUCLEOTIDE sequences for the combinatorial BGC constructs.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python experiments/build_nt_dataset.py

Replaces the back-translation workaround: donors_data.xlsx carries the real
`sequence_nt` per module, so GenomeOcean (a nucleotide model) can be fed genuine
DNA instead of a synthetic reverse translation.

The recipe, established empirically rather than assumed
------------------------------------------------------
`"GG".join(module sequence_aa)` reproduces **all 16,019** protein sequences in
`dataset/constructed_sequence_dataset.csv` byte-for-byte. That pins the recipe
exactly, and shows the `upstream_Architecture` / `downstream_Architecture`
columns are NOT concatenated: they are descriptive labels ('CAT', 'TCLAT', ...),
and each module's stored sequence already spans its flanking architecture.

The nucleotide build mirrors it, with a Gly-Gly codon pair for the "GG" linker:

    "GGCGGC".join(module sequence_nt)

Two repairs to donors_data.xlsx, both verified:
  * 31 modules have spurious 'L' characters inserted into `sequence_nt` (462 in
    total). For every one of them `len_nt - 3*len_aa` equals exactly the number
    of 'L's, and deleting them makes all 31 translate EXACTLY to `sequence_aa`.
    So they are text corruption and deletion is the correct fix.
  * 168 modules have a small `len_nt != 3*len_aa` offset (mostly -1/+2/+3; 149 of
    them still translate to a string starting with the annotated protein). These
    are annotation offsets, not corruption, and the DNA is used as-is.

Why nothing is dropped: the defects are strongly label-correlated. Constructs
touching an 'L' module have a 2.2 % success rate versus 17.2 % for fully clean
ones, so discarding them would push the positive rate from 14.1 % to 17.2 % and
silently bias the task. All 16,019 constructs are kept.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from Bio.Seq import Seq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "dataset")
OUT_DIR = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")

AA_LINKER = "GG"        # two glycines, from the half-T-domain splice
NT_LINKER = "GGCGGC"    # Gly-Gly; GGC is the most-used Gly codon in E. coli
                        # ~30 nt per construct out of ~22 kb, i.e. 0.14 %


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    donors = pd.read_excel(os.path.join(DATA, "donors_data.xlsx"))
    constructs = pd.read_excel(os.path.join(DATA, "experimental_combinatorial_data.xlsx"),
                               sheet_name="Constructs")
    protein_ref = pd.read_csv(os.path.join(DATA, "constructed_sequence_dataset.csv"))
    assert len(constructs) == len(protein_ref) == 16019
    assert (constructs["produced"].values == protein_ref["produced"].values).all(), \
        "Constructs sheet and constructed_sequence_dataset.csv are not row-aligned"

    key = donors["module number"].astype(str)
    aa = dict(zip(key, donors["sequence_aa"].astype(str)))
    # Repair: delete the spurious 'L' characters (verified above).
    nt_raw = donors["sequence_nt"].astype(str)
    n_L = int(nt_raw.str.count("L").sum())
    nt = dict(zip(key, nt_raw.str.replace("L", "", regex=False)))
    print(f"donors: {len(donors)} modules | deleted {n_L} spurious 'L' characters "
          f"from {int((nt_raw.str.count('L') > 0).sum())} modules")

    leftover = {c for s in nt.values() for c in s} - set("ACGTN")
    assert not leftover, f"non-nucleotide characters remain in sequence_nt: {leftover}"

    prot_out, nt_out, n_mod = [], [], []
    for mstr in constructs["modules"]:
        mods = [m.strip() for m in str(mstr).split(",")]
        prot_out.append(AA_LINKER.join(aa[m] for m in mods))
        nt_out.append(NT_LINKER.join(nt[m] for m in mods))
        n_mod.append(len(mods))

    # --- acceptance test 1: protein recipe reproduces the existing dataset ----
    exact = sum(a == b for a, b in zip(prot_out, protein_ref["sequence"]))
    print(f"\n[check] protein recipe reproduces constructed_sequence_dataset.csv: "
          f"{exact}/{len(prot_out)} exact")
    assert exact == len(prot_out), "protein recipe no longer reproduces the reference"

    # --- acceptance test 2: the nt build translates back to that protein -----
    # Only holds where every module's own nt->aa is exact; elsewhere the module
    # annotation itself is offset, so a mismatch is expected and harmless.
    clean_mod = {m for m in nt
                 if len(nt[m]) == 3 * len(aa[m])
                 and str(Seq(nt[m]).translate(table=11)) == aa[m]}
    print(f"[check] modules whose nt translates exactly to aa: {len(clean_mod)}/{len(nt)}")

    n_check = n_ok = 0
    for mstr, prot in zip(constructs["modules"], prot_out):
        mods = [m.strip() for m in str(mstr).split(",")]
        if not all(m in clean_mod for m in mods):
            continue
        n_check += 1
        dna = NT_LINKER.join(nt[m] for m in mods)
        if str(Seq(dna).translate(table=11)) == prot:
            n_ok += 1
    print(f"[check] fully-clean constructs whose reconstructed DNA translates back "
          f"to the verified protein: {n_ok}/{n_check} exact")
    assert n_ok == n_check, "nucleotide reconstruction does not translate to the protein"

    # --- write --------------------------------------------------------------
    out = pd.DataFrame({
        "sequence": nt_out,
        "produced": constructs["produced"].astype(bool),
    })
    aa_len = pd.Series([len(p) for p in prot_out])
    nt_len = pd.Series([len(s) for s in nt_out])
    print(f"\nnucleotide length: median {int(nt_len.median())}, "
          f"p99 {int(nt_len.quantile(0.99))}, max {int(nt_len.max())}")
    est_tok = nt_len / 5.0
    over = int((est_tok > 10240).sum())
    print(f"estimated tokens (at ~5 bp/token): median {int(est_tok.median())}, "
          f"max {int(est_tok.max())} | exceeding the 10240 limit: {over} "
          f"({100 * over / len(out):.2f} %)")
    print(f"positive rate: {out['produced'].mean():.4f} (unchanged, nothing dropped)")

    # The big table goes to scratch: /e/project1/scifi is over its data soft
    # quota and near its inode limit, and this is ~350 MB of sequence.
    path = os.path.join(OUT_DIR, "constructed_nt_dataset.parquet")
    full = out.assign(id=range(len(out)), modules=constructs["modules"].values,
                      n_modules=n_mod, protein=prot_out, aa_len=aa_len, nt_len=nt_len)
    full.to_parquet(path, index=False)
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    print("  columns:", list(full.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
