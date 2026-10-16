# Harder-split experiments — index

Three attempts to build a stricter evaluation than the original
leave-one-family-out split, run in August 2026. Read this file first; each
numbered file is self-contained.

| File | Scheme | Verdict |
|---|---|---|
| [01_jaccard_capped_split.md](01_jaccard_capped_split.md) | Cap max Jaccard similarity to training (0.7 / 0.6 / 0.5) on the existing folds | **The measurement was essential; the capped tables are not.** It produced the study's most important finding — but through the Jaccard *statistic*, not the capped test sets, which change no conclusion and add a bias of their own. |
| [02_multifamily_split.md](02_multifamily_split.md) | Hold out three random families at once | **Failed at its goal.** Widened the test set, did not deepen it. |
| [03_genus_split.md](03_genus_split.md) | Hold out every family from one bacterial genus | **Worked, but small.** 7 folds, 3 primary; the two dominant genera cannot be tested. |

Plus one follow-up analysis that is not a split experiment:

| File | Question | Verdict |
|---|---|---|
| [04_junction_hypothesis.md](04_junction_hypothesis.md) | A colleague's proposal that the 18.3 % one-swap flip rate is multiplicative junction biology rather than noise | **Their arithmetic is right, their test comes back negative.** Internal swaps flip 18.26 %, terminal 18.30 % (model predicts 1.68×); production rate does not decay with length. The simple multiplicative model is ruled out — but their junction-descriptor modelling idea is worth building. |

## Background: what came before

- [`../REPORT.md`](../REPORT.md) — the original random-split experiment. Its test
  set is ~93 % contaminated (see below); its numbers are not family-transfer
  numbers.
- [`../FAMILY_SPLIT.md`](../FAMILY_SPLIT.md) — how the 29 leave-one-family-out
  folds were built.
- [`../FAMILY_RESULTS.md`](../FAMILY_RESULTS.md) — technical results on those
  folds, including the six issues found along the way.
- [`../FAMILY_GUIDE.md`](../FAMILY_GUIDE.md) — the same material written for a
  reader with no background. Start here if the terms below are unfamiliar.

## Read the Jaccard file for the diagnostic, not for the split

Of the three schemes, the Jaccard work is the one that mattered — but not in the
way it was proposed. **Capping similarity changes no conclusion**: within-fold
method rankings are essentially identical to the full test set (median Spearman
rho +1.00 / +0.80 / +0.90 at the three tiers), GenomeOcean is last either way, and
the only rank changes are shuffles inside the already-tied cluster of
Modules / Modules+bigram / Thesis.

It also **adds a bias of its own**. Max-Jaccard turns out to be correlated with
the label — mean |r| = 0.156, above 0.1 on 9 of 12 folds — and the sign flips
between folds (FldS +0.31, XfaS −0.25). So capping does not merely remove
contamination; it reshapes *which kinds* of construct are tested, differently per
fold. XfaS's positive rate more than doubles under the cap (0.103 to 0.233) while
FldS's drops by a quarter. The filter is label-blind in construction but not in
effect.

**What to do instead:** report **max-Jaccard per fold as a strictness
diagnostic** alongside the twin rate, and keep the full test set as the primary
evaluation. The capped tables belong in an appendix as a robustness check
("the ranking survives removing near-duplicates"), not as a headline.

## The one thing to know before reading any of them

**The one-swap "twin rate" is not a reliable measure of how strict a fold is.**

It counts test constructs having a training construct that differs by exactly one
module *at the same length, in the same positions*. It is blind to insertions and
deletions, so a fold whose test constructs have a different module-count
distribution from training is scored as clean regardless of how similar they
really are.

Three folds across three independent split schemes were scored at a 0.0 % twin
rate and are in fact among the most contaminated in the study:

| Scheme | Fold | twin rate | median Jaccard to nearest training construct | length mismatch |
|---|---|---:|---:|---:|
| family | XfaS | 0.0 % | 0.62 | 0.68 |
| genus | Erwinia | 0.0 % | 0.62 | 0.72 |
| multifamily | XfaS+banan | 0.0 % | — | 0.68 |
| | *median, all other folds* | | | *0.31* |

Across all 62 folds, twin rate correlates **−0.46** with test/train length
mismatch. On every one of those three folds the sequence-based methods appear to
beat the module baselines — and on every one, that result disappears once
similarity is measured set-wise instead. **Use the Jaccard measure, not the twin
rate, to judge how strict a fold is.**

## What survives all three schemes

1. **The GenomeOcean-4B embedding is the weakest method tested**, on the family
   split, the genus split, and every Jaccard tier. The single exception is the 10
   primary multi-family folds, where all methods converge — and it is last again
   across all 26 of them.
2. **Methods that know only which modules are present stay competitive
   everywhere.** Nothing tested beats them by a margin that survives a paired
   comparison.
3. **Making the test stricter does not rescue the sequence-based methods.** That
   was the central hypothesis of this whole line of work, and it did not hold.

## Reproducing

```bash
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
cd /e/project1/scifi/thoeppe/genomeoceanbgc

# build the two new split schemes
python experiments/prepare_group_splits.py --mode multifamily
python experiments/prepare_group_splits.py --mode genus

# run every method on a scheme (SPLIT_NAME defaults to `family`)
for M in genus multifamily; do
  sbatch --export=ALL,SPLIT_NAME=$M experiments/family_eval.sbatch
  sbatch --export=ALL,SPLIT_NAME=$M experiments/family_symbolic.sbatch
  sbatch --export=ALL,SPLIT_NAME=$M experiments/family_thesis.sbatch
  sbatch --export=ALL,SPLIT_NAME=$M experiments/family_simple.sbatch
  # ensemble needs eval + thesis first: --dependency=afterok:<eval>:<thesis>
  sbatch --export=ALL,SPLIT_NAME=$M experiments/family_ensemble.sbatch
done

# gather per-fold AP/AUC with bootstrap CIs, then the Jaccard tiers
python experiments/family_collect.py --split-name genus
python experiments/family_jaccard.py --split-name genus
```

All jobs are **CPU-only** — the classifier heads are scikit-learn and the
symbolic model is numpy/scipy, none of which have a GPU backend. **Never run them
on a login node**; they saturate every core they are given.

`--split-name family` is the default everywhere and reproduces the original runs
unchanged.

## Files produced

```
splits/{multifamily,genus}_splits.csv          fold assignments (family, id, split)
splits/{multifamily,genus}_splits_summary.csv  per-fold sizes and diagnostics
results/{family,genus,multifamily}/
    <fold>_{test.csv,scores.npz,...}           per-fold, per-method raw output
    all_methods_ci.csv                         AP + 95 % CI + ROC-AUC, every method
    jaccard_test.csv                           the three Jaccard tiers
    dedup_test.csv                             (family only) the deduplicated tier
```
