# NRPS construct-production splits — v1

Everything needed to benchmark a model on our NRPS construct dataset **on the same
folds, with the same metrics, without the leakage that makes a random split
meaningless.** Self-contained: no access to our code needed.

- **15,985 constructs**, each an artificial chimera of NRPS modules taken from ~29
  natural biosynthetic gene clusters (BGCs).
- **Task:** binary — does the construct produce its compound? Overall rate **14.1 %**.
- **Both DNA and protein sequences included**, so you can run your own encoder.
- **Contact:** tobias.hoeppe@helmholtz-munich.de

---

## Quickest path

```bash
# 0. check the harness reproduces our published numbers on your machine
python -c "
import pandas as pd
b = pd.read_csv('baselines/family_test_scores.csv.gz')
b[b.method == 'thesis'][['fold','id','score']].to_csv('replay.csv', index=False)"
python evaluate.py --pred replay.csv --name thesis_replay --primary-only --n-boot 200
#    -> median_ap 0.380  median_lift 3.829  median_roc_auc 0.826  median_p50 0.500
#    (matches baselines/family_reference_ap.csv; ~20 s)

# 1. run the baseline you are trying to beat (no sequences needed, ~10 min)
python run_all_folds.py --primary-only
python evaluate.py --pred baseline_predictions.csv --primary-only

# 2. pick a fold and export its sequences
python export_fasta.py --kind protein --fold TxlS --split train
python export_fasta.py --kind protein --fold TxlS --split test --no-labels

# 3. encode + fit a head on train, predict test  (your code)
# 4. write predictions as fold,id,score  and score them
python evaluate.py --pred my_scores.csv --primary-only --name my_encoder
```

`evaluate.py` needs only numpy, pandas and scikit-learn.

---

## 1. Why not just split randomly

This dataset is a **combinatorial library**: constructs are built by mixing and matching
modules, so many pairs differ by a single module. That makes a random split almost
useless as a test of generalisation:

| | |
|---|---|
| test constructs with a one-module twin in training (random split) | **93 %** |
| AP of "copy the label of the most similar training construct" | **0.534** |
| AP reported by the original model on that random split | 0.734 |

Most of that 0.734 is recoverable by memorising near-duplicates. On the
module-disjoint folds in this folder the same methods land at **AP 0.28–0.41**. That
gap is the whole reason this folder exists.

One number worth knowing before you start: **18.3 %** of all one-module-swap pairs in
this dataset carry *opposite* labels. So even a perfect model cannot treat near-twins
as interchangeable — the label genuinely moves. (We tested whether this reflects
multiplicative junction biology; it does not follow the gradient that model predicts.)

## 2. How the folds were built

**Leave-one-family-out.** Each of the ~29 natural BGC families becomes one fold:

- **test** = every construct that uses at least one module from the held-out family
- **train** = constructs using no module from that family
- **val** = a deterministic subset of *other* families, held out for model selection
  (never the test family, so tuning cannot see the test distribution)

Fold assignment is deterministic (`zlib.crc32` of the family name, not Python's
salted `hash()`), so the folds are byte-identical on any machine.

### Folds are separate evaluations, not a partition

A construct is a chimera and can draw modules from several families, so it appears in
the test set of **every** fold whose held-out family it touches:

| test rows total | 39,219 |
|---|---|
| distinct constructs appearing as a test row | 15,777 of 15,985 |
| constructs that are a test row in 1 / 2 / 3 / 4+ folds | 2,084 / 6,463 / 4,995 / 2,235 |

**Consequences you must respect:**

1. **Never pool test rows across folds** into one AP. Report per fold and aggregate
   with a median (`evaluate.py` does this).
2. **Fit one model per fold**, on that fold's train split only. There is no single
   train set.

### Use the 12 primary folds

12 of the 29 folds are flagged `primary` in `folds/family_folds_summary.csv` — those
with enough test constructs and positives for a stable AP. The other 17 are reported
for completeness but individually noisy. `evaluate.py --primary-only` selects them:

`FitS · FldS · GxpS · KolS · PdpS · PxbS · SzeS · TxlS · XbcS · XenS · XfaS · XisS`

---

## 3. ⚠️ If you trained your own encoder, read this

This is the one thing that can silently invalidate the whole comparison.

**The split is only module-disjoint if your encoder is too.** If your encoder was
pretrained (even self-supervised, even without labels) on sequences that include a
fold's test constructs, then information about the held-out family is already baked
into the representation. The fold boundary still holds for your *classifier head*, but
not for the model as a whole, and the resulting number is not comparable to ours.

Three ways to handle it, best first:

1. **Pretrain outside this dataset.** If your encoder saw only public genomes/BGCs and
   never these 15,985 constructs, you are clean — just say so. This is the easy case
   and probably yours.
2. **Pretrain per fold** on that fold's `train` split only. Correct, but 29 encoders.
3. **Pretrain once on all constructs.** Fastest, but then report it as such and treat
   the number as an upper bound, not a like-for-like result.

**For reference, what we did:** our `emb_mean` numbers use **frozen**
GenomeOcean-4B-bgcFM embeddings (3072-d, mean-pooled) with a logistic head fitted per
fold. The encoder was never finetuned on this dataset, so only the head sees fold data.
If you finetune end-to-end you will likely beat it, and the comparison should note that
the setups differ.

Two smaller things:

- **Don't train on the `val` split** if you want our numbers to be comparable — we used
  it for model selection, then refit on train+val before scoring test once.
- `export_fasta.py` puts `produced=` in the header by default because it is convenient
  for the train split. Pass `--no-labels` when exporting a test set.

---

## 4. What is in the folder

```
constructs.csv                        15,985 rows — the light metadata table
sequences.parquet                     DNA + protein (116 MB)
folds/
  family_folds.csv                    fold, id, split          ← the main split
  family_folds_summary.csv            per-fold metadata + the `primary` flag
  genus_folds.csv / _summary.csv      harder variant: hold out a whole genus (7 folds)
  multifamily_folds.csv / _summary    hold out 3 families at once (26 folds)
difficulty/
  family_test_difficulty.csv          per (fold, test construct) difficulty labels
  genus_… / multifamily_…             same, for the other schemes
baselines/
  family_reference_ap.csv             our per-fold AP / ROC-AUC / CI
  family_test_scores.csv.gz           our per-construct test scores (paired tests)
  family_copytwin_ap.csv              the leakage baseline, per fold
  family_jaccard_capped_ap.csv        our AP under the J ≤ 0.7 / 0.6 / 0.5 caps
  family_dedup_ap.csv                 our AP after removing one-swap twins
baseline.py                           the baseline in ~45 lines, one CSV in
run_all_folds.py                      the same baseline across every fold
evaluate.py                           scoring harness
export_fasta.py                       parquet → FASTA
MANIFEST.json                         counts, for a quick integrity check
```

### Column dictionaries

**`constructs.csv`** — `id` (join key, int), `modules` (comma-separated module ids, in
order — the construct's DNA is a deterministic function of this list), `n_modules`,
`produced` (bool, the label), `aa_len`, `nt_len`, `random_split` (the *old* contaminated
split; kept only to reproduce our earlier report — **do not use it for evaluation**).

**`sequences.parquet`** — `id`, `sequence` (DNA, median ~22 kb), `protein`, plus the
same metadata columns. Read a subset with
`pd.read_parquet('sequences.parquet', columns=['id','protein'])`.

**`folds/*_folds.csv`** — `family` (the fold name), `id`, `split` ∈ {train, val, test}.
One row per (fold, construct); every construct appears exactly once per fold.

**`folds/*_folds_summary.csv`** — per fold: sizes and positive rates for each split,
`test_modules_unseen_in_train`, `one_swap_leak_pct`, `max_jaccard_median`,
`val_families`, and `primary`.

**`difficulty/*_test_difficulty.csv`** — the useful one for analysis. Per (fold, test
construct):

| column | meaning |
|---|---|
| `fold`, `id`, `produced`, `n_modules` | keys and label |
| `max_jaccard_to_train` | max Jaccard of this construct's module **set** against every training construct in that fold. **This is the strictness measure to use.** Set-based, so it catches substitutions, insertions, deletions and reorderings in one number. |
| `has_one_swap_twin_in_train` | a training construct of the same length differing in exactly one position |
| `n_modules_unseen_in_train` | how many of its modules never occur in that fold's training set |

All three are computed from module identities only — **never from labels** — so
stratifying or filtering on them cannot leak.

**`baselines/family_test_scores.csv.gz`** — `fold, id, y, arm, method, score` (313,752
rows). `arm` is `tuned` (that fold's own validation winner) or `frozen` (the
hyperparameters selected once on the random split). Higher score = more likely to
produce; scores are not calibrated probabilities. `copytwin` is **not** here — it has
no per-construct score file, only the per-fold table.

---

## 5. Reading the metrics

**AP is not comparable across folds.** Chance-level AP equals the fold's positive rate,
which ranges from **0.036 to 0.226** here. So always look at `lift = AP / chance_AP`
alongside raw AP, and aggregate with a per-fold median, never a pooled AP.

**Use `max_jaccard_to_train`, not the twin flag, to judge strictness.** The twin flag is
positional: it only sees a same-length one-position substitution, and misses a training
construct that differs by an insertion or deletion. `XfaS` is the cautionary case — its
twin rate is 0.0 %, which made it look like our cleanest fold, but by Jaccard it is the
*least* clean primary fold (median 0.62). Across 62 folds the twin rate correlates
−0.46 with the train/test length mismatch, i.e. it largely measures whether the fold's
constructs happen to be the same length. Median max-J over the family folds is **0.500**;
42.4 % of test rows have a one-swap twin.

**The Jaccard caps are a stress test, not a headline.** `evaluate.py` reports AP on the
J ≤ 0.7 / 0.6 / 0.5 subsets. A model leaning on near-duplicates fades as the cap
tightens. Two honest caveats: the cap correlates weakly with the label (mean |r| 0.156,
sign varies by fold), and on our methods it changed no conclusion — the ranking was
essentially unchanged (Spearman ρ +0.8 to +1.0). Treat it as a robustness check.

**`prec_at_50`** is the wet-lab-relevant number: of the 50 constructs your model ranks
highest, what fraction actually produce. Against a ~10–14 % base rate our best methods
reach 50–90 % on the primary folds, which is the practical case for using a model at all.

---

## 6. Numbers to beat

Median over the **12 primary family folds** (median chance AP **0.103**):

| method | median AP | median lift | median ROC-AUC | what it is |
|---|---|---|---|---|
| `modules+bigram` | **0.408** | 3.53 | 0.813 | which modules + which adjacent pairs |
| `modules` | 0.407 | 3.61 | 0.813 | which modules, order-blind |
| `simple` | 0.398 | 3.19 | — | modules+bigram + 4 hand-picked numeric features |
| `ensemble` | 0.398 | 3.76 | — | rank-mix of modules+bigram and thesis |
| `thesis` | 0.380 | **3.83** | **0.826** | the original master-thesis pipeline (XGBoost, no foundation model) |
| `symbolic` | 0.322 | 2.67 | 0.805 | hashed 6-mer logistic regression on protein |
| `emb_mean` | 0.278 | 2.34 | 0.761 | **frozen GenomeOcean-4B-bgcFM embedding** + logistic head |
| `copytwin` | 0.258 | 1.97 | — | leakage baseline: copy the nearest training label |

Figures above are the `frozen` arm. Per-fold values, CIs and the `tuned` arm are in
`baselines/family_reference_ap.csv`.

**What this table says.** The five module-based methods are **statistically tied** — the
paired bootstrap cannot separate them. The genome foundation model is **last of the real
methods on every fold and every threshold we tried**, below a plain
which-modules-are-present indicator. Beating `modules+bigram` at AP ≈ 0.41 with a
sequence encoder would be a genuinely new result; that is the bar. `run_all_folds.py`
reproduces it from `constructs.csv` alone — no sequences, no embeddings — in about ten
minutes, and matches the per-fold AP in `baselines/family_reference_ap.csv` exactly on
all 12 primary folds, so you can confirm the bar on your own machine before trying to
clear it. `baseline.py` is the same model stripped to ~45 lines, reading a single CSV
with `modules`, `produced` and `split` columns — start there if you just want to see
what the baseline *is*.

Two structural reasons it is a hard bar, worth knowing before you interpret a low score:

- Construct DNA is a **deterministic function** of the ordered module list (verified:
  15,985 module strings ↔ 15,985 distinct sequences, bijective). So a sequence encoder
  has no information the module indicators lack — it has to *use* it better.
- Indicator features can **abstain** on an unseen module; a sequence model must
  extrapolate. That is exactly what these folds demand, and it cuts both ways.

### The hardest folds

Ranked by median max-Jaccard to training (lower = less similar = harder):

| fold | med max-J | twin % | n test | positives | chance | copytwin | emb_mean | modules+bigram | thesis |
|---|---|---|---|---|---|---|---|---|---|
| `TxlS` | 0.20 | 24.2 | 2,389 | 326 | 0.136 | 0.230 | 0.361 | 0.409 | 0.363 |
| `XenS` | 0.22 | 31.2 | 871 | 78 | 0.090 | 0.361 | 0.073 | 0.413 | 0.380 |
| `XbcS` | 0.33 | 41.1 | 2,347 | 198 | 0.084 | 0.290 | 0.188 | 0.269 | 0.358 |
| `FldS` | 0.33 | 35.6 | 505 | 52 | 0.103 | 0.671 | 0.632 | 0.792 | 0.790 |

`TxlS` and `XenS` are the two we would report first. Note `XenS`: the embedding scores
**0.073**, below its 0.090 chance rate, while module indicators get 0.413 — that fold is
the sharpest illustration of the extrapolation problem.

---

## 7. `evaluate.py` reference

```
--pred FILE          CSV with fold,id,score[,method]   (required)
--scheme {family,genus,multifamily}                    default family
--name STR           label when --pred has no method column
--primary-only       restrict to the 12 trustworthy folds
--compare METHOD     baseline for the paired bootstrap  default modules+bigram
                     ('none' to skip)
--n-boot INT         bootstrap resamples                default 2000
--out FILE           per-fold output table
```

It warns rather than fails when your predictions do not cover a fold completely, and it
exits if it finds duplicate ids. Every metric is rank-based, so scores need no
calibration. The paired bootstrap scores you and the baseline on the **same** resampled
rows, which is what makes small differences interpretable; `folds_significant` counts
folds where the paired 95 % CI excludes zero.

## 8. Other schemes

`genus` (7 folds, 3 primary) holds out every family from one bacterial genus — stricter,
but the two dominant genera cannot be tested, so it is small. `multifamily` (26 folds)
holds out three families at once; **it failed at its goal** — 98.9 % of its test
constructs still touch only one of the three held-out families, so it widened the test
set without deepening it. Included for completeness; **`family` is the one to use.**

## 9. Known limitations

- **12 usable folds** is not many. Differences below ~0.03 AP are generally not
  resolvable, which is why the module-based methods come out tied.
- Fold difficulty varies a lot (median max-J 0.20 to 0.62); a model can look good by
  doing well on easy folds. Check the per-fold table, not just the median.
- `symbolic` was run with hyperparameters frozen from the random split, with no per-fold
  sweep (29 × 37 sparse fits was too expensive). Compare it to other methods' `frozen`
  arm only; a stale hyperparameter cannot be ruled out as the cause of its weaker score.
- Labels come from experimental production assays and carry real noise; the 18.3 %
  one-swap flip rate is a lower bound on how much the label moves for near-identical
  designs.
