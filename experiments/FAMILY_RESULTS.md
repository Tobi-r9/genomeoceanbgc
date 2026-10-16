# Predicting BGC product formation across biosynthetic families

REPORT.md's evaluation, re-run on the 29 leave-one-family-out folds built by
[prepare_family_splits.py](prepare_family_splits.py) and documented in
[FAMILY_SPLIT.md](FAMILY_SPLIT.md).

**REPORT.md and the random-split pipeline are untouched.** Everything here lives
in new files and writes to new paths; the original runs remain reproducible.

## Result

**The parts-list baselines did not collapse. Nothing beats them, and the
GenomeOcean embedding is the weakest method of the five.**

REPORT.md predicted that a module-disjoint split is "exactly the setting where
the parts-list baselines should collapse and a sequence model could pull ahead."
Across the 12 primary folds, the opposite happened. The order-blind bag of
modules — no sequence, no embedding, no pretrained model — is statistically tied
with the best method on the board, and beats the embedding outright.

The one method that matches the parts lists is the **thesis method**, a hybrid of
indicator and sequence features. It is tied with them on AP, degrades least as
more of the test construct becomes unseen, and beats `emb_mean` significantly on
9 of 12 folds.

Median across the 12 primary folds (lift = AP ÷ that fold's positive rate):

| Arm | | `emb_mean` | `symbolic` | `thesis` | `modules+bigram` | `modules` |
|---|---|---:|---:|---:|---:|---:|
| **tuned** | median AP | 0.329 | — | — | **0.395** | 0.375 |
| | median lift | 3.00 | — | — | 3.05 | **3.74** |
| **frozen** | median AP | 0.278 | 0.322 | 0.380 | **0.408** | 0.407 |
| | median lift | 2.34 | 2.67 | **3.83** | 3.53 | 3.61 |
| | folds won (of 12) | 0 | 1 | **5** | 4 | 2 |

`thesis` is the master-thesis method from `NRPS-Models-and-data`, added later; see
[The thesis method](#the-nrps-models-and-data-thesis-method). Like `symbolic` it has
a frozen arm only. Adding it changes the frozen folds-won counts, since it wins
folds the others previously did.

For reference, on the random split (REPORT.md) every method sat at lift 4.9–5.2
with AP 0.694–0.734. **Everything drops under a family split — but the embedding
drops furthest.**

Median AP and median lift disagree on which baseline leads because the median of
a ratio is not the ratio of medians; the folds differ enormously in prevalence
(chance AP 0.036 to 0.226). Lift is the comparable quantity.

## What was run

Per fold, mirroring `train_classifiers.py` + `test_table.py`:

1. sweep all 19 classifier configurations per feature block, fit on TRAIN, score VAL;
2. **tuned** arm — that fold's own validation winner; **frozen** arm — the
   configuration REPORT.md selected on the random split, refit unchanged;
3. refit the chosen config on TRAIN+VAL, score TEST exactly once;
4. 2,000-resample percentile bootstrap CI on test AP, plus a paired bootstrap of
   each baseline against `emb_mean` on the same resampled rows.

Validation is itself family-held-out, so nothing is tuned on a random draw.

`symbolic` has **no tuned arm** — its configuration is frozen from REPORT.md, as
agreed, because a per-fold sweep would be 29 × 37 slow sparse fits. It is
therefore comparable to the other methods' frozen arm, **not** their tuned arm.

## Primary folds, tuned arm

| Family | leak | chance | `emb_mean` | `modules+bigram` | `modules` | winner |
|---|---:|---:|---:|---:|---:|---|
| GxpS | 11.5 % | 0.146 | **0.489** | 0.479 | 0.437 | emb |
| TxlS | 24.2 % | 0.136 | 0.378 | 0.384 | **0.402** | modules |
| XenS | 31.2 % | 0.090 | 0.103 | **0.418** | 0.346 | bigram |
| FitS | 34.5 % | 0.170 | 0.332 | 0.331 | **0.348** | modules |
| FldS | 35.6 % | 0.103 | 0.701 | **0.792** | 0.780 | bigram |
| XisS | 36.6 % | 0.036 | 0.098 | 0.153 | **0.176** | modules |
| PdpS | 38.2 % | 0.096 | 0.313 | **0.600** | 0.594 | bigram |
| SzeS | 39.1 % | 0.159 | **0.448** | 0.407 | 0.412 | emb |
| XbcS | 41.1 % | 0.084 | 0.154 | 0.159 | **0.344** | modules |
| KolS | 42.5 % | 0.155 | 0.542 | 0.436 | **0.562** | modules |
| PxbS | 44.0 % | 0.068 | 0.253 | 0.264 | **0.264** | tie |
| **XfaS** | **0.0 %** | 0.103 | **0.327** | 0.197 | 0.251 | emb |

Paired bootstrap on test, `emb_mean` minus each baseline:

| Arm | vs | folds emb wins | median ΔAP | emb significantly better |
|---|---|---:|---:|---:|
| tuned | `modules` | 3 / 12 | −0.023 | 3 |
| tuned | `modules+bigram` | 5 / 12 | −0.006 | 3 |
| frozen | `modules` | 1 / 12 | −0.103 | 0 |
| frozen | `modules+bigram` | 1 / 12 | −0.108 | 1 |

## Retuning matters, and almost entirely for the embedding

This is why both arms were worth running. Going from the frozen REPORT.md config
to per-fold selection moves `emb_mean`'s median lift 2.34 → 3.00, while the
module baselines barely move (3.61 → 3.74 and 3.53 → 3.05). Individual folds
swing hard: GxpS 0.286 → 0.489, PdpS 0.140 → 0.313, PxbS 0.144 → 0.253.

The reason is visible in what the sweep picks. On the random split the winner was
`logreg C=10`. On the family folds, `emb_mean` selects **much stronger
regularisation**:

| Selected for `emb_mean` (12 primary folds) | folds |
|---|---:|
| `logreg C=0.001` | 3 |
| `logreg C=0.01` | 2 |
| `logreg C=0.1` | 2 |
| MLP variants | 3 |
| gradient boosting | 2 |

Seven of twelve folds pick a C at least 100× smaller than the random split's
choice. **A meaningful part of what would look like "the embedding fails to
transfer" is a stale hyperparameter**, and any future comparison that reuses
REPORT.md's config on a shifted split will understate the embedding.

## The two folds that matter most

**GxpS (GameXPeptide)** — the headline fold: largest test set (6,212, 39 % of the
data), 906 positives, and the lowest leakage of any large fold at 11.5 %.

| Arm | Method | Test AP | 95 % CI | ROC-AUC |
|---|---|---:|:---:|---:|
| tuned | `emb_mean` (MLP 256×64) | **0.489** | [0.455, 0.524] | 0.849 |
| tuned | `modules+bigram` (HGB) | 0.479 | [0.448, 0.513] | 0.818 |
| tuned | `modules` (HGB) | 0.437 | [0.407, 0.470] | 0.795 |
| frozen | `symbolic` | 0.374 | — | 0.832 |
| frozen | `emb_mean` (logreg C=10) | 0.286 | [0.265, 0.309] | 0.762 |

Paired bootstrap, tuned arm: versus `modules+bigram` ΔAP +0.010 [−0.022, +0.041],
**not significant**; versus `modules` ΔAP +0.051 [+0.020, +0.082], **significant**.
So on the cleanest large fold the embedding beats the order-blind parts list but
is indistinguishable from parts-list-plus-order. Chance AP is 0.146, so 0.489 is
a 3.35× lift — against 5.2× on the random split.

**XfaS** — the only fold with 0.0 % residual leakage, the strictest test in the set:

| Method | tuned AP | frozen AP | ROC-AUC (frozen) |
|---|---:|---:|---:|
| `emb_mean` | **0.327** | 0.313 | 0.751 |
| `symbolic` | — | 0.314 | **0.790** |
| `modules` | 0.251 | 0.258 | 0.762 |
| `modules+bigram` | 0.197 | 0.208 | 0.730 |

Here the embedding does pull ahead, and by the largest relative margin anywhere
(3.18× lift versus 1.92× and 2.45×). It is also the fold where the module
baselines have least to work with. **This is the single strongest piece of
evidence for the foundation model in the whole study — and it rests on 545 test
constructs with 56 positives, so it is suggestive, not conclusive.**

## Where the `symbolic` baseline works, and where it breaks

`symbolic` is the weakest-looking method in the summary table, so it is worth
being precise about when it fails. **It works on most folds.** Comparison is on
the frozen arm throughout, the only fair one, since `symbolic` has no tuned arm.

Across all 29 folds it is:

- the **best of the four methods on 7 folds** — PdpS, BacS, RirS, SldS, AmbS, XfaS, XldS;
- **better than the GenomeOcean embedding on 21 of 29 folds**;
- at **lift >= 3 on 16 of 29 folds**;
- a genuine failure on only 4 — XenS, XbcS, FitS and SypS (lift < 2).

The folds where it leads, all frozen-arm lift over chance:

| Fold | `symbolic` | `emb_mean` | `modules+bigram` | `modules` | test positives |
|---|---:|---:|---:|---:|---:|
| PdpS | **6.60** | 1.45 | 6.48 | 6.02 | 93 |
| BacS | **5.68** | 2.98 | 2.76 | 3.71 | 42 |
| RirS | **4.65** | 2.07 | 4.11 | 3.74 | 260 |
| SldS | **4.19** | 3.44 | 3.30 | 2.89 | 196 |
| AmbS | **3.92** | 3.21 | 3.79 | 3.60 | 54 |
| XfaS | **3.06** | 3.05 | 2.03 | 2.51 | 56 |
| XldS | **2.70** | 2.66 | 2.20 | 2.58 | 233 |

XfaS is the most interesting of these: it is the only fold with 0.0 % residual
leakage, i.e. the strictest test in the whole study, and there **both
sequence-reading methods beat both parts-list baselines** (`symbolic` 3.06 and
`emb_mean` 3.05, against 2.51 and 2.03).

### One variable explains the failures

For each fold, measure what fraction of an average test construct's modules
belong to the held-out family. Correlating that with test ROC-AUC on the primary
folds separates the methods sharply:

| Method | r(ROC-AUC, fraction of construct held out) |
|---|---:|
| `symbolic` | **−0.78** |
| `emb_mean` | −0.42 frozen / −0.59 tuned |
| `modules` | −0.27 frozen / −0.03 tuned |
| `modules+bigram` | −0.21 frozen / −0.28 tuned |

**The two sequence-reading methods degrade sharply as more of the test construct
is unseen; the two parts-list methods barely notice.** All five folds where at
least a third of the construct is held out (XenS 0.46, TxlS 0.45, XbcS 0.40,
FitS 0.36, SzeS 0.33) sit in the bottom eight by symbolic lift.

XenS is the extreme, and it is not a symbolic-specific problem:

| XenS (46 % of each test construct held out) | ROC-AUC |
|---|---:|
| `modules` | 0.865 |
| `modules+bigram` | 0.850 |
| `symbolic` | 0.413 |
| `emb_mean` | **0.360** |

Both sequence methods score **below 0.5**, i.e. they rank producers worse than
chance, while both parts-list methods are unaffected.

### Why: indicator features abstain, sequence features extrapolate

For XenS the 8 held-out modules appear in **zero** training constructs. In the
parts-list multi-hot their columns are therefore all-zero during training, the
classifier learns no weight for them, and at test time they contribute nothing —
the prediction falls back on the **3.29 modules per construct that are still
known**. The model degrades gracefully into a partial parts list.

A sequence model cannot do that. Those residues are physically present in the
test protein: XenS test constructs average 6,305 residues against 5,029 in
training. Amino-acid frequencies, hashed 6-mer counts and the mean-pooled
embedding all shift into a region the model never saw, and there is no "unknown"
bucket to fall into, so it extrapolates — and extrapolation can invert the
ranking.

`symbolic` is hit hardest because part of its feature set is explicitly
module-identity memorisation: the first-50 and last-50 residue signatures, which
REPORT.md notes "largely identify the first and last module". On XenS only 34 %
of test prefixes and 34 % of suffixes were ever seen in training, and 63 % of
test constructs have a held-out first module. Prefix coverage correlates +0.56
with symbolic's ROC-AUC across the primary folds.

**This asymmetry is intrinsic to the benchmark, not a defect in any one method.**
Part of what the parts-list baselines are being rewarded for is the ability to
*abstain* on unseen chemistry, rather than superior understanding of it. That
should be read alongside the headline result, since it cuts against the premise
that a family split would automatically favour a sequence model.

Not changed here: `symbolic` is left exactly as REPORT.md ran it. Two untested
levers remain if it is ever worth revisiting — giving it the per-fold tuning the
other methods got (worth +0.66 median lift to `emb_mean`, and its `l2=3.0` was
likewise chosen on the easy split), and dropping the terminal signatures, which
are provably dead weight on held-out modules. The second would make it a
different method.

## The NRPS-Models-and-data thesis method

Added after the four methods above, on the same 29 folds: the pipeline from
`NRPS-Models-and-data/Recent_Models/model.py`, reimplemented in
[family_thesis.py](family_thesis.py) against our dataset.

Its features are a **hybrid**: organism one-hot (an indicator, one column per
organism seen in the fitting set), architecture-compatibility counts along the
chain, GC statistics capped at 65 %, and TF-IDF over character 3-5-grams of both
the nucleotide and the protein sequence (1000 features each). Then StandardScaler
and XGBoost with their exact parameters — `binary:logistic`, depth 6, eta 0.05,
subsample 0.9, colsample 0.9, `hist`, 300 rounds, `scale_pos_weight` from the
class ratio.

### Result: it beats both sequence methods decisively and ties the parts lists

Paired bootstrap on test, 2,000 resamples, 12 primary folds, `thesis` minus each
comparator — the rigorous comparison, done on AP directly:

| vs | thesis better on | significantly better | median ΔAP |
|---|---:|---:|---:|
| `emb_mean` | **11 / 12** | **9** | **+0.152** |
| `modules+bigram` | 5 / 12 | 3 | −0.007 |
| `modules` | 6 / 12 | 4 | −0.004 |

**Read the median lift with care.** `thesis` has the highest median lift of any
method (3.83 against 3.61 and 3.53) and wins the most folds, but on raw AP the
paired test says it is **statistically tied with both parts-list baselines** —
median ΔAP is −0.007 and −0.004, i.e. essentially zero. Lift is a ratio and
weights the low-prevalence folds more heavily; the paired bootstrap on AP is the
more trustworthy statement. What survives is: **`thesis` beats `emb_mean`
(9 of 12 folds significantly) and `symbolic`, and draws with the parts lists.**

Per-fold, with 95 % bootstrap CIs on the thesis AP:

| Fold | thesis AP | 95 % CI | ΔAP vs `modules+bigram` | ΔAP vs `modules` |
|---|---:|:---:|---:|---:|
| FldS | 0.790 | [0.688, 0.872] | −0.000 | +0.030 |
| KolS | 0.574 | [0.487, 0.658] | −0.014 | −0.024 |
| GxpS | 0.482 | [0.447, 0.517] | +0.006 | **+0.046** |
| SzeS | 0.471 | [0.432, 0.512] | **+0.063** | **+0.058** |
| XfaS | 0.467 | [0.336, 0.613] | **+0.256** | **+0.206** |
| PdpS | 0.380 | [0.302, 0.483] | −0.234 | −0.195 |
| XenS | 0.380 | [0.289, 0.502] | −0.027 | −0.039 |
| TxlS | 0.363 | [0.315, 0.419] | −0.044 | −0.038 |
| XbcS | 0.358 | [0.298, 0.428] | **+0.089** | **+0.071** |
| FitS | 0.356 | [0.308, 0.408] | −0.039 | −0.020 |
| PxbS | 0.289 | [0.195, 0.413] | +0.026 | +0.012 |
| XisS | 0.087 | [0.059, 0.148] | −0.090 | −0.083 |

Bold = significant (paired CI excludes zero). It wins big on XfaS and loses big
on PdpS; net, a draw.

On the 17 secondary folds it looks stronger still — median lift 4.35, winning 10
of 17 — but those folds are leaky and are not the basis for any claim here.

### It is the most robust method to unseen chemistry

Correlating each method's test ROC-AUC with the fraction of the average test
construct that belongs to the held-out family:

| Method | r(ROC-AUC, fraction held out) | median AUC |
|---|---:|---:|
| **`thesis`** | **−0.19** | **0.826** |
| `modules+bigram` | −0.21 | 0.813 |
| `modules` | −0.27 | 0.813 |
| `emb_mean` | −0.42 | 0.761 |
| `symbolic` | −0.78 | 0.805 |

It degrades least of all five, and it does so while carrying sequence features —
which is the interesting part. The two folds that made this concrete earlier:

| | XenS (46 % held out) | XfaS (0.0 % leakage, strictest) |
|---|---:|---:|
| `thesis` | **0.380** (AUC 0.851) | **0.467** (AUC 0.800) |
| `modules` | 0.424 (0.865) | 0.258 (0.762) |
| `modules+bigram` | 0.413 (0.850) | 0.208 (0.730) |
| `symbolic` | 0.077 (0.413) | 0.314 (0.790) |
| `emb_mean` | 0.073 (0.360) | 0.313 (0.751) |

On XenS it holds up where both pure-sequence methods fall **below chance**; on
XfaS it is significantly the best of all five.

### Why the hybrid survives

The organism one-hot behaves exactly like the module multi-hot: an organism only
present in the held-out family has an all-zero column during training, so it
contributes nothing at test time rather than pushing the model somewhere it has
never been. The GC and architecture-compatibility features are **family-agnostic
numeric summaries** — "how many adjacent module pairs are architecturally
compatible", "what is the mean GC" — which stay on the same scale no matter which
modules appear. Only the TF-IDF block has to extrapolate.

So this method gets the abstention property of an indicator model *and* whatever
sequence signal is available, which is what neither the pure parts-list nor the
pure sequence methods manage. That, rather than XGBoost, is the plausible reason
it transfers best.

### Deviations from their code

1. **TF-IDF, the scaler and the organism vocabulary are fitted on the fitting set
   only.** `model.py` fits the vectorizers on its training block, but the sibling
   notebooks fit them on the whole dataset before splitting, which leaks. The
   non-leaky reading is used, matching every other method here.
2. **Our protocol.** Their code has no validation set — it splits train/test by
   row position. Here: fit on TRAIN and score VAL, then refit on TRAIN+VAL and
   score TEST once.
3. **No hyperparameter sweep**, because their code has none. `thesis` therefore
   has a frozen arm only and belongs in the frozen column. Its draw with the
   parts lists and its win over `emb_mean` are achieved *without* the per-fold
   tuning the other methods got, which if anything understates it.
4. **`module_architecture` is the architecture column.** Their donor sheet had one
   `Module_Architecture`; ours has `module_architecture` (16 values) and
   `Module_Architecture_uncommon` (44). Measured over all 71,159 adjacent module
   pairs, the match feature fires for 42.4 % / 42.9 % of pairs with the former and
   only 0.2 % / 2.0 % with the latter, so `_uncommon` would have been a silently
   dead feature.
5. Their **substring** organism test is reproduced as written, not fixed:
   `Cystobacter sp.` also matches `Cystobacter sp. SBCb004`. It is their method.

`xgboost` 3.4.0 was installed into the environment for this (aarch64 wheel, no
dependency changes — `pip install --dry-run` confirmed numpy, scipy and nccl were
already satisfied).

## Confusion tables and balanced subsets

Produced by [family_confusion.py](family_confusion.py) for all 12 primary folds,
using the model REPORT.md committed to (`emb_mean` + logreg C=10) — which, per
the section above, is now known to be a poor choice on these folds. Read these as
"what the committed model does under family shift", not as the method's ceiling.

GxpS, threshold 0.5 and the F1-optimal validation threshold (0.139):

| Set | thr | n | pos | TN | FP | FN | TP | prec | rec | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.5 | 2,118 | 177 | 1,705 | 236 | 119 | 58 | 0.197 | 0.328 | 0.246 | 0.165 |
| test | 0.5 | 6,212 | 906 | 2,836 | 2,470 | 82 | **824** | 0.250 | **0.909** | 0.392 | 0.314 |
| test | 0.139 | 6,212 | 906 | 2,549 | 2,757 | 63 | 843 | 0.234 | 0.930 | 0.374 | 0.294 |

The class-weighted model becomes extremely liberal under shift: it recovers 91 %
of producers at 25 % precision, flagging 3,294 of 6,212 constructs. Compare the
random split, where the same model got 82 % recall at 55 % precision.

Balanced 50/50 subsets were written for all 12 primary folds
(`splits/family_equal/<fold>_{val,test}_equal.csv`). As in REPORT.md they make
accuracy interpretable and inflate precision/AP purely through prevalence — e.g.
FldS test-equal reaches accuracy 0.769 and AP 0.960 on 104 rows.

## Secondary folds

The 17 folds with residual leakage above 45 % are in
`results/family/summary_test.csv` and behave the same way, only more so: median
lift `emb_mean` 3.44 versus `modules+bigram` 4.17 and `modules` 3.78 in the tuned
arm. They are reported separately and deliberately **not** averaged in with the
primary folds — a fold where 96 % of test constructs still have a near-twin in
training (SypS) is not measuring family transfer.

## Issues found

Six, all of which affected this work.

**1. The `symbolic` input CSVs cannot be regenerated from the repo.**
`run_symbolic.py` reads `$GO_SCRATCH/bgc_clf/symbolic/{train,val,trainval,test}.csv`,
but no script in the repo writes them; they exist on scratch from an uncommitted
manual step of 6 August. REPORT.md's `symbolic` numbers therefore cannot be
reproduced from a clean checkout. Worked around by rebuilding equivalent frames
in memory from `dataset_split.parquet`; the originals were not touched.

**2. REPORT.md's frozen `symbolic` configuration does not transfer.**
`n_kmers=150000` requires 150,000 hashed 6-mer bins to clear `min_df=5`, and
three folds cannot supply that many — the folds that hold out a large family
leave a smaller, less diverse training set. `family_symbolic.py` now caps
`n_kmers` at what is available and records both numbers. The count is label-blind
(it sums document frequencies over both classes), so this is a capacity fallback,
not a tuning decision.

| Fold | eligible bins | `n_kmers` used | test AP |
|---|---:|---:|---:|
| GxpS | 125,266 | 125,266 | 0.374 |
| PxbS | 141,898 | 141,898 | 0.189 |
| XabS | 148,949 | 148,949 | 0.471 |

**3. A failing fold killed its whole shard.** In the first symbolic job, GxpS and
XabS raised the error above. Because each was at position 0 of its shard, the 5
folds queued behind them were never attempted — 7 folds missing from 2 real
failures. Both drivers now wrap each fold in `try/except`, log the failure, and
exit non-zero at the end, so a bad fold can no longer silently truncate a run.

**4. `confusion_and_balanced.py` overwrites `splits/val_equal.csv` and
`splits/test_equal.csv`.** Reusing it for the family folds would have destroyed
the random-split versions. `family_confusion.py` writes to
`splits/family_equal/` instead.

**5. Two existing scripts pick the F1-optimal threshold differently.**
`confusion_and_balanced.py` uses `argmax(f1)` over the full precision-recall
curve, which would index `thresholds` one past its end if the maximum landed on
the final point; `run_symbolic.py` uses the guarded `argmax(f1[:-1])`. They agree
here because that final point has recall 0, but the guarded form is used
throughout this work since some folds have as few as 30 test positives.

**6. A bug in this work's own aggregator**, found and fixed: `summary_test.csv`
matched the `*_test.csv` input glob, so a second run of `family_tables.py`
re-ingested its own output and double-counted every fold. The AP tables were
unaffected (duplicates averaged to identical values) but the metadata merge
broke. Now filtered by filename prefix.

## Caveats

- **`symbolic` has no tuned arm.** Where it underperforms, a stale
  hyperparameter cannot be ruled out — exactly the effect that costs `emb_mean`
  0.66 of median lift. See
  [Where the `symbolic` baseline works](#where-the-symbolic-baseline-works-and-where-it-breaks):
  it leads on 7 of 29 folds and beats the embedding on 21 of 29.
- **The comparison partly rewards abstention, not chemistry.** Parts-list models
  can zero out an unseen module; sequence models must extrapolate over residues
  that are physically in the input. That asymmetry, not superior understanding,
  explains a large part of why the parts-list baselines win — and it is the main
  reason to be cautious about reading the headline as "the foundation model does
  not transfer".
- **Folds are not independent.** Every fold trains on an overlapping subset of
  the same 15,985 constructs, so the 12 primary results are not 12 independent
  replicates and their spread understates the true uncertainty.
- **Small positive counts.** Test positives range from 30 (Gra) to 906 (GxpS);
  folds at the low end have AP intervals far wider than any gap reported here.
- Residual leakage is 0–44 % even on the primary folds. Only XfaS and XtvB are
  essentially clean.
- Test-set sizes vary from 320 to 6,212 constructs, and training sets from 7,655
  to 13,068, so the GxpS result is additionally penalised by having the least
  training data.
- No per-fold re-embedding was needed or done; the frozen GenomeOcean embeddings
  are split-independent.

## Reproduce

```bash
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
cd /e/project1/scifi/thoeppe/genomeoceanbgc

python experiments/prepare_family_splits.py    # 29 folds (already built)
sbatch experiments/family_eval.sbatch          # 3 methods x 29 folds, ~53 min
sbatch experiments/family_symbolic.sbatch      # symbolic x 29 folds, ~40 min
sbatch experiments/family_thesis.sbatch        # thesis method x 29 folds, ~36 min
sbatch experiments/family_confusion.sbatch     # confusion + balanced, ~4 min
python experiments/family_tables.py            # aggregate + print all tables
```

All three jobs are CPU-only and **must not be run on a login node** — the heads
are scikit-learn and the symbolic model is numpy + scipy, none of which have a
GPU backend, and this environment has no xgboost, lightgbm, cuML or cupy. The
sbatch scripts request no `--gres` for that reason; booster nodes are exclusive,
so all 288 cores are allocated either way. All three are re-runnable and skip
folds that already have output.

Raw results: `results/family/summary_test.csv`, `summary_paired.csv`,
`confusion_tables.csv`, and per-fold `<fold>_{val_sweep,test,paired}.csv`,
`<fold>_symbolic.json`, `<fold>_scores.npz`.
