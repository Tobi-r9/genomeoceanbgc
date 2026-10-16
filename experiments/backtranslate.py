"""Back-translate amino-acid sequences to nucleotides for a DNA-only model.

GenomeOcean's vocabulary is pure ACGTN (4096 BPE tokens, zero entries containing
any other character), so amino-acid input tokenises to ~79 % `[UNK]` and carries
almost no signal. Back-translation is the adapter: one codon per residue.

What this does and does not do
-----------------------------
DNA -> protein is deterministic but many-to-one (2-6 synonymous codons per
residue), so the *true* source nucleotide sequence is NOT recoverable. What is
recoverable is a *plausible* coding sequence carrying exactly the same
information: back-translation is a (seeded) function of the protein, so no
information is created or destroyed, and a downstream classifier can only
exploit protein content. The cost is that codon usage and third-base wobble
become artificial, i.e. somewhat out-of-distribution for a model trained on real
metagenomic DNA.

Two schemes, so the choice can be measured rather than assumed:

``mode="frequency"`` (default)
    Sample synonymous codons in proportion to E. coli K-12 relative usage, with
    a fixed seed. Reproduces realistic codon-usage statistics, keeping the input
    closer to the model's training distribution.

``mode="most_frequent"``
    Always emit the single most-used codon per residue. Fully deterministic, but
    gives unnaturally uniform codon usage (every Leu becomes CTG).

Special characters in this dataset: ``*`` -> a real stop codon (2454 sequences
carry internal stops, i.e. they are multi-gene constructs, and marking the gene
boundary is meaningful to a genome model); ``X`` -> ``NNN``.
"""
from __future__ import annotations

import numpy as np

# E. coli K-12 codon usage, per-thousand (Kazusa). Only relative weights within
# each synonymous family matter here.
_CODON_USAGE = {
    'A': {'GCG': 33.7, 'GCC': 25.5, 'GCA': 20.2, 'GCT': 15.3},
    'R': {'CGC': 22.0, 'CGT': 20.9, 'CGG': 5.4, 'CGA': 3.6, 'AGA': 2.1, 'AGG': 1.2},
    'N': {'AAC': 21.6, 'AAT': 17.7},
    'D': {'GAT': 32.1, 'GAC': 19.1},
    'C': {'TGC': 6.5, 'TGT': 5.2},
    'Q': {'CAG': 28.8, 'CAA': 15.900},
    'E': {'GAA': 39.4, 'GAG': 17.8},
    'G': {'GGC': 29.6, 'GGT': 24.7, 'GGG': 11.1, 'GGA': 8.0},
    'H': {'CAT': 12.9, 'CAC': 9.7},
    'I': {'ATC': 25.1, 'ATT': 30.3, 'ATA': 4.4},
    'L': {'CTG': 52.6, 'TTA': 13.9, 'TTG': 13.7, 'CTC': 11.1, 'CTT': 11.0, 'CTA': 3.9},
    'K': {'AAA': 33.6, 'AAG': 10.3},
    'M': {'ATG': 27.9},
    'F': {'TTT': 22.360, 'TTC': 16.6},
    'P': {'CCG': 23.2, 'CCA': 8.4, 'CCT': 7.0, 'CCC': 5.5},
    'S': {'AGC': 16.1, 'TCT': 8.5, 'TCC': 8.6, 'TCG': 8.9, 'AGT': 8.8, 'TCA': 7.2},
    'T': {'ACC': 23.4, 'ACG': 14.4, 'ACT': 9.0, 'ACA': 7.1},
    'W': {'TGG': 15.2},
    'Y': {'TAT': 16.3, 'TAC': 12.2},
    'V': {'GTG': 26.4, 'GTT': 18.3, 'GTC': 15.3, 'GTA': 10.9},
    '*': {'TAA': 2.0, 'TGA': 0.9, 'TAG': 0.2},
}

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _validate_table() -> None:
    """The table is hand-entered; assert it agrees with the bacterial code."""
    from Bio.Data import CodonTable
    t = CodonTable.unambiguous_dna_by_id[11]
    for aa, codons in _CODON_USAGE.items():
        for codon in codons:
            if aa == '*':
                assert codon in t.stop_codons, f"{codon} is not a stop codon"
            else:
                got = t.forward_table.get(codon)
                assert got == aa, f"{codon} encodes {got}, not {aa}"
    missing = set(STANDARD_AAS) - set(_CODON_USAGE)
    assert not missing, f"residues missing from usage table: {missing}"


_validate_table()

# Deterministic scheme: the single most-used codon per residue.
_MOST_FREQUENT = {aa: max(c, key=c.get) for aa, c in _CODON_USAGE.items()}

# Sampling scheme: parallel arrays of codons and normalised probabilities.
_CODONS = {aa: np.array(sorted(c)) for aa, c in _CODON_USAGE.items()}
_PROBS = {
    aa: np.array([_CODON_USAGE[aa][c] for c in _CODONS[aa]], dtype=np.float64)
    for aa in _CODON_USAGE
}
for _aa in _PROBS:
    _PROBS[_aa] /= _PROBS[_aa].sum()

_UNKNOWN_CODON = "NNN"


def back_translate(protein: str, mode: str = "frequency",
                   rng: np.random.Generator | None = None) -> str:
    """Return a nucleotide sequence coding for `protein` (3x its length)."""
    if mode == "most_frequent":
        return "".join(_MOST_FREQUENT.get(aa, _UNKNOWN_CODON) for aa in protein)
    if mode != "frequency":
        raise ValueError(f"unknown mode {mode!r}")
    if rng is None:
        raise ValueError("mode='frequency' needs an explicit rng for reproducibility")

    out = []
    for aa in protein:
        codons = _CODONS.get(aa)
        if codons is None:            # 'X' and anything else unexpected
            out.append(_UNKNOWN_CODON)
        elif len(codons) == 1:
            out.append(codons[0])
        else:
            out.append(codons[rng.choice(len(codons), p=_PROBS[aa])])
    return "".join(out)


def back_translate_many(proteins, mode="frequency", seed=0):
    """Back-translate a list of proteins with one seeded rng for the whole batch."""
    rng = np.random.default_rng(seed) if mode == "frequency" else None
    return [back_translate(p, mode=mode, rng=rng) for p in proteins]


if __name__ == "__main__":
    from Bio.Seq import Seq
    rng = np.random.default_rng(0)
    for prot in ("MKVLA*", "ACDEFGHIKLMNPQRSTVWY", "MKXVA"):
        for mode in ("most_frequent", "frequency"):
            dna = back_translate(prot, mode=mode, rng=rng)
            # Round-trip: translating back must reproduce the protein exactly
            # (X/NNN excepted, which Biopython renders as 'X' anyway).
            got = str(Seq(dna).translate(table=11))
            status = "OK" if got == prot else f"MISMATCH {got!r}"
            print(f"{mode:14s} {prot:22s} -> {dna[:36]:36s} len {len(dna):3d}  {status}")
