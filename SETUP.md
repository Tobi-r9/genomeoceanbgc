# Task: predict whether a combinatorial BGC construct produces a product

You are given a dataset of engineered non-ribosomal peptide synthetase (NRPS)
constructs and a binary outcome. Build the best classifier you can and report the
metrics listed at the bottom.

> ### Read this first — do not open these files
>
> A previous attempt at this task exists in this repository. **To keep your result
> independent, do not read any of:**
>
> - `experiments/REPORT.md`
> - `experiments/DATASET_NOTES.md`
> - `experiments/results/` (any file)
> - `experiments/train_classifiers.py`, `experiments/analyze_val.py`,
>   `experiments/confusion_and_balanced.py`, `experiments/embed_dataset.py`
> - `CLAUDE.md` sections about experiments
>
> Everything you need is in this file. `experiments/splits/split.csv` and the
> dataset files listed below **are** fair game — you must use the given split.

## The biology, briefly

An NRPS assembly line is built from **modules** strung together in order. Each
module adds one amino acid to a growing peptide. Researchers combinatorially
recombined modules from a donor library into new constructs and measured, in the
lab, whether each construct actually produced a detectable peptide. Many do not:
junctions between modules from different origins often fail.

Your job: predict that success/failure label from the construct.

## The data

Everything lives under `dataset/`.

**`experimental_combinatorial_data.xlsx`**, sheet `Constructs` — 16,019 rows, the
label source:

| column | meaning |
|---|---|
| `modules` | the construct, as an **ordered, comma-separated list of module numbers**, e.g. `1215,1216,105,1218,1219,1222`. Order is meaningful — it is the order of the assembly line. |
| `produced` | the label, `True` / `False` |
| `peptide` | the intended peptide, e.g. `Leu-Thr-Leu-Tyr-Tyr-Leu` |

The same file has two other sheets you may or may not find useful: `Modules`
(per-module metadata: role, organism, architecture, specificity, GC content) and
`Transitions` (observed success counts for `from_module` → `to_module` pairs).

**`donors_data.xlsx`** — 1,384 rows, one per donor module, 57 columns. The
important ones are `module number` (the join key), `sequence_nt` (the module's
nucleotide sequence), `sequence_aa` (its amino-acid sequence), plus metadata such
as `module_architecture`, `specificity`, `organism`, `cluster`.

**`constructed_sequence_dataset.csv`** — 16,019 rows of `sequence`, `produced`,
where `sequence` is the **amino-acid** sequence of the whole construct, aligned
row-for-row with the `Constructs` sheet.

### Pre-built sequences (recommended starting point)

A validated, ready-to-use table is at:

```
/e/scratch/profound/thoeppe/genomeocean/bgc_clf/constructed_nt_dataset.parquet
```

Columns: `id`, `modules`, `n_modules`, `produced`, `sequence` (the construct's
full **nucleotide** sequence), `protein`, `aa_len`, `nt_len`. Construct sequences
were assembled from the per-module sequences in `donors_data.xlsx` in the order
given by `modules`.

You may rebuild this yourself from the spreadsheets if you prefer, but be warned
that the raw `sequence_nt` column contains some malformed entries that need
handling. Using the parquet removes that variable so that any difference in
results reflects modelling, not data cleaning.

## The split — you must use this one

`experiments/splits/split.csv` assigns every construct to `train`, `val`, or
`test`. It has columns `id`, `split`, `produced`, `n_modules`, `aa_len`, `nt_len`,
where `id` joins to the parquet above.

If you work from the spreadsheets instead, join on the `modules` string, which is
unique per construct — see `experiments/splits/split_by_modules.csv`.

It is a stratified random 70/15/15 split over 15,985 constructs (exact duplicates
removed):

| split | n | positive rate |
|---|---:|---:|
| train | 11,190 | 0.1413 |
| val | 2,398 | 0.1414 |
| test | 2,397 | 0.1410 |

**Protocol.** Do all model choice, hyperparameter tuning and threshold selection
using **train and validation only**. Evaluate on test **once**, at the end. Do not
iterate against the test score.

## Compute available

A ready environment with PyTorch, transformers, vLLM and scikit-learn:

```bash
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
```

`$GO_BGC_MODEL` points at a local copy of **GenomeOcean-4B-bgcFM**, a 4B-parameter
nucleotide (DNA) foundation model fine-tuned on biosynthetic gene clusters. You
are free to use it, ignore it, or use something else entirely. Two things to know
if you do use it: load it with `trust_remote_code=False` (the bundled remote code
is broken on this transformers version), and its context limit is 10,240 tokens at
roughly 5 bp per token.

The machine is JUPITER (aarch64, GH200 GPUs). Login nodes have one GPU and
internet; for more, submit to Slurm with `--account=scifi --partition=booster
--gres=gpu:4`, max 12 h, and note that **compute nodes have no internet**, so
stage anything you download first. Write large files to
`/e/scratch/profound/thoeppe/`, never to `/e/project1` (it is near its quota).

## What to report

The label is imbalanced: **14.1 % positive**. Accuracy is therefore close to
useless — always predicting "not produced" scores 85.9 % — so do not lead with it.

Report, on the **test** set:

1. **Average precision** (area under the precision–recall curve) — the primary
   metric. Note that chance level equals the positive rate, 0.141.
2. **ROC-AUC.**
3. **A confusion matrix**, stating the threshold you used and how you chose it
   (choose it on validation, not test).
4. Precision, recall and F1 at that threshold.

Also report your **validation** numbers for the same, and say briefly what you
tried and what you selected.

Two things that make a result much more informative, if you have the budget:

- **Baselines.** Compare against at least one cheap feature set that uses no
  sequence model, so it is clear whether a large model is earning its place.
- **An estimate of uncertainty** on your test AP, e.g. a bootstrap interval.
  With only ~338 positives in the test set, differences of a few points may not
  be meaningful.

## Known caveat, so you can factor it in

This is a *combinatorial* library: all constructs are assembled from the same pool
of ~1,384 donor modules, so train and test necessarily share modules. A random
split therefore measures interpolation within a known module pool rather than
generalisation to genuinely new chemistry. Interpret your number accordingly, and
say so in your write-up.
