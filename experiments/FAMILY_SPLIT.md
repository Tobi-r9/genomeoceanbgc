# Leave-one-family-out splits for the BGC production task

A second, harder set of train/val/test splits for the same 15,985 constructs.
Each fold holds out **one biosynthetic family** — every donor module that came
from a given wild-type BGC — so the model is asked to transfer to chemistry it
has never seen rather than to interpolate inside a known parts pool.

**Status: the splits are built and checked. No models have been trained on them
yet.** Nothing in the existing pipeline was modified;
[REPORT.md](REPORT.md) and [train_classifiers.py](train_classifiers.py) are
untouched, and no embeddings were recomputed.

Built by [prepare_family_splits.py](prepare_family_splits.py).

## Why the existing split is not enough

The split in [prepare_data.py](prepare_data.py) is a stratified random 70/15/15.
Every construct is assembled from the same 1,384-module pool, so train and test
share modules by construction. Measured on that split:

| Leakage measure | Value |
|---|---:|
| Test constructs whose exact module **set** also appears in train | 12.4 % |
| Test constructs with a train construct differing by exactly **one module in one position** | **93.0 %** |

For nine out of ten test constructs the model has already seen an almost
identical recipe with a known label. That is interpolation within a known pool,
not generalisation to new chemistry — and it is the most likely reason the
parts-list baselines in REPORT.md match the 4B embedding.

## What "family" means here

The family label is the **`cluster` column of `dataset/donors_data.xlsx`**: the
wild-type BGC a donor module was lifted from. `GxpS` is GameXPeptide synthetase,
`SzeS`, `XbcS`, `TxlS` and so on are the other synthetases.

Of 140 distinct cluster labels on the 1,384 donor modules, **116 survive as
usable families** across the 1,173 modules actually used by the 15,985
constructs. The rest are dropped as non-families (below).

This label was chosen over the alternatives in the same spreadsheet because it
is the finest biologically-meaningful grain available. `genus` (46 values) is
coarser and puts more than half the data in one group; `WT` (222 values) is
finer but fragments into too many groups to hold out individually. `organism`
tracks `cluster` closely without adding resolution.

## The split rule

**Module-disjoint, not construct-disjoint.** A construct is a chimera drawing
modules from 3–4 different clusters, so it has no single family label. Holding
out family F therefore means:

- **train** — constructs using *no* module belonging to F
- **test** — constructs using *at least one* module belonging to F

A pure-family test set is not available at this dataset size: only 0–31
constructs are built entirely from a single cluster, depending on the cluster.

Four further decisions:

**Hybrid cluster labels count for both parents.** 26 modules carry labels like
`GxpS/XgsC` or `KolS/GxpS`. These are split on `/` and the module is treated as
a member of every parent family. This is the conservative choice: holding out
GxpS then also removes the hybrids, so no GxpS-derived module survives in
training. It costs little — for GxpS it moves the test set from 5,848 to 6,212
constructs.

**`mix`, `unknown` and blank are not families** and are never held out. They are
bookkeeping values, not biosynthetic origins. Modules carrying them stay in
training. A construct can still land in test if it *also* uses a family-F
module, which is correct — the construct is contaminated either way.

**Validation is itself family-held-out.** Choosing hyperparameters on a randomly
drawn validation set would select the model that interpolates best, which is the
exact failure mode this split exists to avoid. So whole families are also held
out of the training pool to form validation, giving a genuine three-level
hierarchy: **training sees neither the validation families nor the test family.**
Validation families are picked by greedy accumulation over a seeded shuffle
until validation reaches ~15 % of the post-test pool, capped at 30 %. Several
small families are preferred over one large one, so that whether a single family
happens to be easy does not drive every model choice.

**The split is label-blind.** Family membership comes from the donor
spreadsheet. Nothing in the construction touches the `produced` column, an
embedding, or any trained model — so the split cannot be circular with respect
to the thing being predicted.

## Which families become folds

A family is emitted as a fold only if it is worth running:

- test ≥ 200 constructs **and** ≥ 30 positives (otherwise the AP confidence
  interval is wider than any effect worth detecting)
- train ≥ 5,000 constructs
- test ≤ 45 % of the dataset (otherwise the comparison against the random split
  is confounded by a training-set-size effect)

**29 of the 116 families qualify.**

## The folds

`leak` is the one-swap statistic defined above, measured test-against-train for
that fold, and is directly comparable to the 93.0 % of the random split. `unseen`
is the number of modules appearing in test that appear nowhere in training.

| Family | train | val | test | pos train | pos val | pos test | test pos | unseen | leak | primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| GxpS | 7,655 | 2,118 | 6,212 | 0.154 | 0.084 | 0.146 | 906 | 265 | **11.5 %** | ✓ |
| SzeS | 9,848 | 2,075 | 4,062 | 0.139 | 0.117 | 0.159 | 647 | 68 | 39.1 % | ✓ |
| TxlS | 10,440 | 3,156 | 2,389 | 0.135 | 0.164 | 0.137 | 326 | 17 | 24.2 % | ✓ |
| XbcS | 11,528 | 2,110 | 2,347 | 0.150 | 0.159 | 0.084 | 198 | 9 | 41.1 % | ✓ |
| RirS | 10,806 | 3,005 | 2,174 | 0.134 | 0.184 | 0.120 | 260 | 18 | 60.2 % | |
| XabS | 11,375 | 2,538 | 2,072 | 0.147 | 0.058 | 0.214 | 443 | 134 | 50.9 % | |
| FitS | 10,048 | 4,074 | 1,863 | 0.134 | 0.146 | 0.170 | 317 | 25 | 34.5 % | ✓ |
| XtpS | 11,861 | 2,508 | 1,616 | 0.133 | 0.162 | 0.171 | 277 | 90 | 51.8 % | |
| SldS | 12,347 | 2,257 | 1,381 | 0.150 | 0.091 | 0.142 | 196 | 26 | 48.7 % | |
| SymS | 12,534 | 2,269 | 1,182 | 0.155 | 0.093 | 0.088 | 104 | 13 | 87.8 % | |
| Pps | 12,280 | 2,613 | 1,092 | 0.162 | 0.068 | 0.086 | 94 | 25 | 45.1 % | |
| XisS | 11,026 | 3,892 | 1,067 | 0.165 | 0.104 | 0.036 | 38 | 52 | 36.6 % | ✓ |
| XldS | 11,770 | 3,184 | 1,031 | 0.149 | 0.086 | 0.226 | 233 | 114 | 50.1 % | |
| PxbS | 11,217 | 3,771 | 997 | 0.144 | 0.153 | 0.068 | 68 | 29 | 44.0 % | ✓ |
| BacS | 11,209 | 3,780 | 996 | 0.141 | 0.169 | 0.042 | 42 | 48 | 60.4 % | |
| PdpS | 12,398 | 2,620 | 967 | 0.151 | 0.114 | 0.096 | 93 | 16 | 38.2 % | ✓ |
| Bicornutin | 12,118 | 2,927 | 940 | 0.137 | 0.147 | 0.185 | 174 | 38 | 63.5 % | |
| XenS | 12,508 | 2,606 | 871 | 0.154 | 0.097 | 0.090 | 78 | 13 | 31.2 % | ✓ |
| CipS | 12,629 | 2,491 | 865 | 0.149 | 0.133 | 0.057 | 49 | 20 | 82.0 % | |
| KolS | 12,866 | 2,293 | 826 | 0.155 | 0.062 | 0.155 | 128 | 76 | 42.5 % | ✓ |
| Gra | 12,702 | 2,691 | 592 | 0.152 | 0.113 | 0.051 | 30 | 12 | 91.0 % | |
| XfaS | 12,639 | 2,801 | 545 | 0.163 | 0.051 | 0.103 | 56 | 13 | **0.0 %** | ✓ |
| CifS | 12,685 | 2,788 | 512 | 0.131 | 0.184 | 0.176 | 90 | 9 | 91.6 % | |
| FldS | 12,935 | 2,545 | 505 | 0.160 | 0.053 | 0.103 | 52 | 52 | 35.6 % | ✓ |
| SypS | 13,068 | 2,423 | 494 | 0.139 | 0.166 | 0.089 | 44 | 22 | 96.2 % | |
| AmbS | 12,480 | 3,023 | 482 | 0.135 | 0.172 | 0.112 | 54 | 57 | 53.1 % | |
| XtvB | 12,922 | 2,599 | 464 | 0.142 | 0.140 | 0.140 | 65 | 91 | **0.6 %** | |
| BanS | 12,836 | 2,794 | 355 | 0.155 | 0.082 | 0.104 | 37 | 11 | 92.4 % | |
| AB1394 | 12,409 | 3,256 | 320 | 0.142 | 0.142 | 0.106 | 34 | 12 | 76.9 % | |

## Which folds to trust

Clean family bookkeeping does not guarantee a clean test. A family can be
held out perfectly and still leave test constructs that differ from a training
construct by one module, because the *other* modules in those constructs are
shared. Folds are therefore tagged:

- **12 primary folds** — leak ≤ 45 % and test ≥ 500: GxpS, SzeS, TxlS, XbcS,
  FitS, XisS, PxbS, PdpS, XenS, KolS, XfaS, FldS. Median leak **36.1 %**.
- **17 secondary folds** — retained because they are still evidence, but median
  leak **60.4 %**, up to 96.2 % for SypS. Report them separately; do not average
  them in with the primary folds.

Two folds are essentially leak-free: **XfaS at 0.0 %** and **XtvB at 0.6 %**.
XtvB misses the primary tag only on test size (464 < 500) and is worth reporting
regardless — it is the strictest fold in the set.

**GxpS is the headline fold**: largest test set (6,212, 39 % of the data),
906 positives, and the lowest leak of any large fold at 11.5 %. It is also the
family the wet-lab side asked about. Its cost is that training drops to 7,655
constructs, so a fair comparison against the random-split number needs the
reference model retrained on a matched-size sample.

**Every fold needs its own chance line.** Test positive rates span 0.036 (XisS)
to 0.226 (XldS), so AP is not comparable across folds without dividing through
by prevalence. This is the same prevalence effect documented in REPORT.md's
balanced-subset section.

## Output files

`experiments/splits/family_splits.csv` — 463,565 rows, one per (fold, construct):

```
family,id,split
GxpS,0,train
GxpS,1,train
GxpS,2,train
```

29 folds × 15,985 constructs; 343,139 train / 81,207 val / 39,219 test rows in
total. `id` matches `splits/split.csv` and the cached embedding shards, so
**nothing has to be re-embedded** — only the classifier heads are refit.

`experiments/splits/family_splits_summary.csv` — one row per fold with the
columns in the table above plus `max_jaccard_median`,
`test_frac_maxjac_ge_0.7`, `n_val_families` and `val_families`.

## Integrity checks

Printed by the script on every run:

- **Permutation siblings never straddle the boundary: 0 across all 29 folds.**
  Assignment depends only on the module *set*, so the 264 constructs whose
  permutation sibling carries the opposite label always land on the same side.
  The random split does not have this property.
- **Every fold is genuinely module-disjoint for its family** — 9 to 265 modules
  appear in test that appear nowhere in training.
- **Reproducible.** `md5sum` of `family_splits.csv` is identical across repeated
  runs and across different `PYTHONHASHSEED` values. Validation families are
  seeded with `zlib.crc32(family)` rather than Python's `hash()`, which is
  salted per process and would otherwise give a different validation set every
  run.

## Not done yet

- **No models have been trained on these folds.** No numbers in REPORT.md apply
  to them, and REPORT.md has not been edited.
- `train_classifiers.py` has no flag for these splits yet. Wiring it up means
  overriding `df["split"]` from `family_splits.csv` after `load()` and **before**
  `feature_blocks()` — the module-bigram vocabulary is built from the training
  rows, so it has to follow the new boundary.
- When the sweep is run, the four methods from REPORT.md (`emb_mean`,
  `symbolic`, `modules+bigram`, `modules`) should all be run on identical folds.
  The prediction worth testing is that the parts-list baselines collapse while
  the sequence model holds up. Expect every number to fall relative to the
  random split; AP 0.734 is not the target.
