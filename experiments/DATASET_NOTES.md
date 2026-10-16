# Reconstructing nucleotide sequences for the combinatorial BGC constructs

How `dataset/*.xlsx` becomes model input, and the evidence for each step. Built
by [build_nt_dataset.py](build_nt_dataset.py); the split is
[prepare_data.py](prepare_data.py).

## Why this was needed

`dataset/constructed_sequence_dataset.csv` holds **amino-acid** sequences, but
GenomeOcean-4B-bgcFM is a **nucleotide** model: its whole 4096-token vocabulary
is ACGTN (zero entries contain any other character). Tokenising protein with it
yields **79.3 % `[UNK]`** and 1.03 chars/token, against 0 % and 5.13 on real DNA.
The ~21 % that survive are only the residues spelled A/C/G/T/N
(Ala/Cys/Gly/Thr/Asn), read as nucleotides. Embedding the protein file directly
would have encoded mostly noise.

`donors_data.xlsx` carries a real `sequence_nt` per module, so genuine DNA can be
reconstructed instead. (A back-translation fallback exists in
[backtranslate.py](backtranslate.py) and is no longer used — kept because it is
the right tool if a module ever lacks `sequence_nt`.)

## The recipe

```
protein construct     = "GG".join(module sequence_aa)
nucleotide construct  = "GGCGGC".join(module sequence_nt)
```

Modules come from the `modules` column of the `Constructs` sheet of
`experimental_combinatorial_data.xlsx`, in order; the `"GG"` is the Gly-Gly scar
left by splicing modules at half-T-domains.

**The protein form is proven, not assumed:** it reproduces all **16,019** rows of
`constructed_sequence_dataset.csv` byte-for-byte. The nucleotide form mirrors it
and is verified by translation — see below.

### The architecture columns are NOT concatenated

The team's working description was `UA_<module>_DA`, reading
`upstream_Architecture` / `downstream_Architecture` from `donors_data`. Those
columns are **descriptive labels** (`CAT`, `AT`, `TCLAT`, `TECAT`, …; 43 and 49
distinct values), not sequence. Each module's stored `sequence_aa` / `sequence_nt`
**already spans its flanking architecture**, which is why plain
`"GG".join(sequence_aa)` reproduces every construct exactly. Adding UA/DA
sequence would double-count the flanks.

### The Gly-Gly linker in nucleotide space

`"GG"` is unambiguous in protein space but degenerate in DNA. The real cloning
scar is not recorded in the spreadsheets, so `GGCGGC` is used (GGC is the
most-used Gly codon in *E. coli*). This is ~30 nt per construct out of ~15 kb,
i.e. **0.14 %** — immaterial to embeddings, but it is a choice, not a fact. If the
true scar is known, change `NT_LINKER` in `build_nt_dataset.py`.

## Two defects in `donors_data.xlsx`, both repaired

**1. Spurious `L` characters in `sequence_nt`** — 462 of them across 31 modules,
e.g. module 1 contains `CATTC` **`L`** `ATTGC`. `L` is not a nucleotide.

Evidence they are insertions to delete: for **every** affected module,
`len_nt - 3*len_aa` equals *exactly* the number of `L`s, and after deleting them
all **31/31 translate exactly** to the annotated `sequence_aa`. So this is text
corruption, and deletion is the correct repair — not masking to `N`, which would
leave the reading frame shifted.

**2. Small length offsets** — 168 modules have `len_nt != 3*len_aa` (mostly −1,
+2, +3; 149 of them still translate to a string starting with the annotated
protein). These are annotation offsets rather than corruption, and the DNA is
used **as-is**: a ±2 nt offset in a ~3.2 kb module does not matter for embeddings.

After repair, **1215 of 1384** modules translate exactly to their annotated
protein (up from 1184).

## Nothing is dropped, because the defects are label-correlated

| Construct group | n | positive rate |
|---|---|---|
| all modules clean | 11,508 | **17.21 %** |
| touches an `L` module | 2,380 | **2.23 %** |
| touches a length/translation-offset module | 2,131 | 10.65 % |
| **overall** | **16,019** | **14.11 %** |

Discarding the defective groups would have moved the positive rate from 14.11 %
to 17.21 % and silently changed the task. All 16,019 constructs are kept. (The
2.23 % success rate of `L`-module constructs is striking and may be a real
biological or batch effect — worth a look independently of this pipeline.)

## Verification

`build_nt_dataset.py` asserts all three on every run:

1. Protein recipe reproduces `constructed_sequence_dataset.csv`: **16,019/16,019** exact.
2. Modules whose `sequence_nt` translates exactly to `sequence_aa`: **1215/1384**.
3. Constructs built purely from those clean modules whose reconstructed DNA
   translates back to the verified protein: **13,514/13,514** exact.

Check 3 is the strong one: it closes the loop from spreadsheet to DNA to protein
against an independently verified reference.

## Resulting dataset

`$GO_SCRATCH/bgc_clf/constructed_nt_dataset.parquet` (122 MB) — columns
`sequence` (nucleotide), `produced`, plus `id`, `modules`, `n_modules`,
`protein`, `aa_len`, `nt_len`. It lives on scratch, not in `dataset/`, because
`/e/project1/scifi` is over its data soft quota and near its inode limit.

- 16,019 constructs, 14.11 % positive.
- Nucleotide length: median **14,814**, p99 27,428, max 73,558.
- Tokenised: median **2,969** tokens (~5 bp/token), and only **5 constructs
  (0.03 %)** exceed GenomeOcean's 10,240-token limit and get truncated.

## Split

`experiments/splits/split.csv`, seed `20260805`, stratified 70/15/15.

Deduplication is on the **nucleotide** sequence, i.e. exactly what the model
sees. That matters: there are only **32** duplicate rows and **2** label
conflicts at DNA level, versus **933** and **237** at protein level — those 237
"conflicts" are *different* constructs that happen to encode the same peptide, so
they are legitimate distinct examples, not noise. Working in DNA space therefore
keeps 15,985 of 16,019 rows (99.8 %) instead of 14,849 (92.7 %).

| split | n | negatives | positives | positive rate |
|---|---|---|---|---|
| train | 11,190 | 9,609 | 1,581 | 0.14129 |
| val | 2,398 | 2,059 | 339 | 0.14137 |
| test | 2,397 | 2,059 | 338 | 0.14101 |

Max positive-rate deviation across splits: **0.00036**.

> **Caveat, by construction.** This is a *combinatorial* library: constructs are
> built from a shared pool of 1,384 modules, so train and test necessarily share
> modules and are far from independent. A random stratified split therefore gives
> an **optimistic** estimate of generalisation to genuinely new chemistry. The
> `modules` multi-hot baseline in
> [train_classifiers.py](train_classifiers.py) exists to quantify how much of any
> score is module identity rather than sequence understanding. A
> module-disjoint or cluster-based split would answer the harder question.

The peptide-level conflicts are themselves informative: identical peptides with
different outcomes means the peptide sequence alone does not determine success.
