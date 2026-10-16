# Predicting BGC product formation from frozen GenomeOcean-4B-bgcFM embeddings

Predict the binary `produced` label for a combinatorial NRPS construct.
15,985 constructs, **14.1 % positive**. The 4B model is **frozen** — only a small
classifier is trained on its embeddings.

## Result

**Test AP 0.734, ROC-AUC 0.929** (chance AP = 0.141, so a 5.2× lift).
Validation was 0.729 / 0.925, so nothing overfitted.

At threshold 0.5: **finds 82 % of producers at 55 % precision**. Random picking
would give 14 % precision.

**But three methods with no pretrained model match it on test.** A contributed
hand-engineered classifier (`symbolic`) reaches AP 0.724, a baseline that knows
only which modules a construct contains and in what order (`modules+bigram`)
reaches 0.708, and the order-blind parts list alone (`modules`) reaches 0.694 —
against the embedding's 0.734. **No gap is statistically significant**, and on
ROC-AUC the two module baselines actually score *higher* than the embedding. See
[Final test table](#final-test-table--all-four-methods) and
[Baselines](#baselines--does-the-4b-model-earn-its-place) before drawing
conclusions about the foundation model.

## The model

```
construct DNA (median 14.8 kb)
  -> GenomeOcean-4B-bgcFM, frozen, bf16          # forward pass only, no training
  -> mean-pool last hidden state over tokens     # 3072-dim vector
  -> StandardScaler
  -> LogisticRegression(C=10, class_weight="balanced")
```

That is the whole thing. The 4B model is never updated; embeddings are computed
once and cached, and only the logistic regression is fitted (~20 s on CPU).

**The input is the 3072 embedding numbers and nothing else** — no module
information, no length, no composition. I also tested a variant that glued the
module parts-list onto the embedding (`emb_mean+modules`, 3072 + 1173 = 4245
inputs). It scored a hair higher, 0.735 vs 0.729, but that gap is far smaller
than the measurement noise (see [bootstrap CIs](#how-to-read-the-95--ci)), so I
kept the simpler model that uses the embedding alone.

## Where the data comes from

`dataset/constructed_sequence_dataset.csv` contains **protein** sequences, which a
nucleotide model cannot read (79 % `[UNK]`). So the DNA is assembled from the
spreadsheets instead:

- `donors_data.xlsx` gives a real `sequence_nt` **per module** (1,384 modules).
- `experimental_combinatorial_data.xlsx` gives each construct as an **ordered list
  of ~6 module numbers**, plus the `produced` label.
- Construct DNA = those module sequences joined in order, with a `GGCGGC`
  (Gly-Gly) linker at each junction.

So "reconstruction" just means stitching module-level DNA into construct-level
DNA. Verified: the same recipe in protein space reproduces all 16,019 rows of the
existing CSV byte-for-byte, and the assembled DNA translates back to that protein
for all 13,514 constructs whose modules are clean. Two spreadsheet defects were
repaired first — details in [DATASET_NOTES.md](DATASET_NOTES.md).

Split: stratified 70/15/15 (11,190 / 2,398 / 2,397), positive rate 0.1413 /
0.1414 / 0.1410.

## Baselines — does the 4B model earn its place?

Each row below is a **different input** fed to the same pool of classifiers
(logistic regression, small MLPs, gradient boosting — 19 configurations each).
The score shown is the best classifier for that input, **on validation** — this
is the table that chose the models. Test numbers for the four headline methods
are in the [final test table](#final-test-table--all-four-methods).

| Input | What the classifier sees | Best head | Val AP | 95 % CI | ROC-AUC |
|---|---|---|---:|---|---:|
| **`emb_mean`** | **3072 embedding numbers** | **LogReg C=10** | **0.729** | [0.680, 0.778] | 0.925 |
| `modules+bigram` | parts list **plus module order** (see below) | grad. boosting | 0.712 | [0.659, 0.760] | **0.926** |
| `symbolic` | hand-designed protein features + hashed 6-mers (see below) | its own LogReg | 0.698 | — | 0.924 |
| `modules` | 1173 zeros/ones: which donor modules are in this construct. No sequence, no order. | grad. boosting | 0.683 | [0.627, 0.736] | 0.923 |
| `modules_bigram` | order (adjacent pairs) only, without the parts list | MLP(256,64) | 0.683 | [0.629, 0.734] | 0.903 |
| `emb_last` | 3072 numbers, last-token instead of mean-pooled | MLP(64) | 0.682 | [0.631, 0.731] | 0.912 |
| `aac` | 20 amino-acid fractions + length | grad. boosting | 0.655 | [0.601, 0.706] | 0.897 |
| `len` | sequence length + module count (3 numbers) | grad. boosting | 0.342 | [0.297, 0.391] | 0.764 |
| chance | — | — | 0.141 | — | 0.500 |

### The parts-list baselines, concretely

There are 1,173 distinct donor modules. Each construct is an ordered list of ~6 of
them, e.g. `1215, 1216, 105, 1218, 1219, 1222`.

**`modules`** turns that into a row of 1,173 numbers, 0 everywhere except a 1 in
the slot of each module present — about 6 ones and 1,167 zeros. That row is the
entire input; the classifier never sees a nucleotide. It asks: can you predict
success purely from *which parts were used*?

That encoding is **order-blind**, which is a real defect here: 2,214 constructs
(13.8 %) share their module set with another construct, and for 264 of them
(1.65 %) the permutation siblings carry **opposite labels**, so a bag-of-modules
model provably cannot fit both. For example, swapping two adjacent modules flips
the outcome:

```
1114,114,111,10,107,2102  -> produced       1104,103,101,175,1992 -> produced
1114,114,10,111,107,2102  -> not produced   1104,101,103,175,1992 -> not produced
```

**`modules+bigram`** fixes this by adding one feature per adjacent pair
`module_i -> module_{i+1}` (5,407 pairs seen in training, plus `^`/`$` sentinels
marking the first and last module). This is the same notion as the `Transitions`
sheet of `experimental_combinatorial_data.xlsx`. Vocabulary is built on train
only.

Order is worth **+0.029 AP** (0.683 → 0.712) — real, and it matters for how the
headline should be read.

### The `symbolic` method (contributed)

`symbolic_nrps_logistic.py` is a standalone, hand-engineered classifier
contributed by a collaborator, run here unmodified on our split via
[run_symbolic.py](run_symbolic.py). It works on the **protein** sequence and uses
no pretrained model at all. Its features are:

- **39 handcrafted measurements** — log length, length and length², `GG` count
  (which effectively counts module junctions), `*` and `X` counts, the fraction
  of non-standard characters, the 20 amino-acid frequencies, and counts of 12
  known NRPS domain motifs (`SSGTT`, `YKTGDL`, `GGHSL`, …).
- **Hashed 6-mer log-counts** — every overlapping 6-residue window hashed into
  2²⁰ bins, of which the most class-discriminative are kept. Selection uses a
  `(pos_df − neg_df)² / (total + 10)` score computed on **training labels only**.
- **Terminal signatures** — the first 50 and last 50 residues as categorical
  indicators, plus their pair, keeping values seen at least twice in training. In
  a combinatorial library these largely identify the **first and last module**.

Then L2-regularised logistic regression (L-BFGS) on the sparse matrix.

Run on our split it needed tuning to be represented fairly. At its shipped
defaults it scores **val AP 0.663**, but L-BFGS stops on the iteration cap with
`success: false` — an under-fit model rather than a converged one. Sweeping 37
configurations (motif count, L2, iteration budget) on train/val gives a converged
optimum at **`--n-kmers 150000 --l2 3.0 --max-iterations 2500`** and
**val AP 0.698**. Motif count plateaus there (80k → 150k gains 0.001).

**Test AP 0.724, ROC-AUC 0.924** — remarkably close to the embedding's 0.734 /
0.929. Paired bootstrap on test: ΔAP +0.009, CI [−0.031, +0.049],
P(embedding better) = 0.67. **Indistinguishable.** A hand-designed feature set with
no pretrained model matches the 4B foundation model on this task.

### How to read the 95 % CI

The AP of 0.729 was measured on one particular set of 2,398 validation
constructs. With a different 2,398 constructs the number would have come out
somewhat different, and the interval says how much.

Since we cannot collect fresh validation sets, we fake them from the one we have.
Draw 2,398 constructs at random *with replacement* from the 2,398 we own — some
get picked twice, some not at all — and compute AP on that. Repeat 2,000 times.
Sort the 2,000 results and take the 2.5th and 97.5th percentiles: 95 % of them
land between 0.680 and 0.778.

Practical reading: **the measurement is only good to about ±0.05 AP**, so any two
models closer together than that cannot be told apart with this much data. That
is why the 0.006 gap between `emb_mean` and `emb_mean+modules` was ignored.

Paired bootstrap against `emb_mean`, **on validation**. "Paired" means both models
are scored on the *same* resampled constructs each round and we record the
difference; that cancels out easy and hard draws, making it more sensitive than
comparing two intervals by eye. If 95 % of the 2,000 differences are above zero,
the gap is real:

| vs | ΔAP | 95 % CI | Verdict |
|---|---:|---|---|
| `len` | +0.386 | [+0.324, +0.439] | **significant** |
| `aac` | +0.074 | [+0.021, +0.127] | **significant** |
| `emb_last` | +0.047 | [+0.000, +0.090] | marginal |
| `modules` | +0.046 | [−0.004, +0.096] | not significant |
| `symbolic` | +0.032 | [−0.010, +0.073] | not significant |
| `modules+bigram` | **+0.017** | **[−0.027, +0.061]** | **clearly not significant** |

**This is the main negative result, and it should be read before the headline.**
`modules+bigram` knows only which modules a construct contains and in what order —
no nucleotides, no embedding, no pretrained model, just a list of parts and
junctions. It matches the 4B model's ROC-AUC exactly (0.926 vs 0.925) and trails
its AP by 0.017, well inside the noise (P(better) = 0.77, i.e. close to a coin
flip).

So on this dataset and this split, **the GenomeOcean embedding is not demonstrably
better than knowing the parts list and their order.** It beats composition-level
features convincingly, but that is a much weaker claim. The
[test evaluation](#final-test-table--all-four-methods) confirms this: the gap
stays inside the noise there too (+0.025 AP, CI [−0.016, +0.066]).

Also tried, all within noise (≤ +0.006 AP): L2-normalising the embedding, and
concatenating it with length, composition, or the module vector.

## Final test table — all four methods

Every model refit on **train+val** with the configuration that won on validation,
then scored **once** on the 2,397 test constructs (338 positive, 14.1 %). No
tuning happens here. CIs are 2,000-resample percentile bootstraps over test rows.

| Method | Test AP | 95 % CI | Test ROC-AUC |
|---|---:|:---:|---:|
| `emb_mean` | **0.734** | [0.685, 0.780] | 0.929 |
| `symbolic` | 0.724 | [0.676, 0.771] | 0.924 |
| `modules+bigram` | 0.708 | [0.659, 0.755] | **0.934** |
| `modules` | 0.694 | [0.642, 0.741] | 0.931 |
| chance | 0.141 | — | 0.500 |

Paired bootstrap against `emb_mean`, same resampled rows each round:

| vs | ΔAP | 95 % CI | P(emb better) | Verdict |
|---|---:|---|---:|---|
| `modules` | +0.040 | [−0.003, +0.083] | 0.96 | not significant |
| `modules+bigram` | +0.025 | [−0.016, +0.066] | 0.88 | not significant |
| `symbolic` | +0.009 | [−0.031, +0.049] | 0.67 | not significant |

Two things to note. **The intervals overlap almost entirely** — all four methods
are within measurement noise of each other. And **the ROC-AUC ordering flips**:
the module baselines beat the embedding there. That is not a contradiction. AP
weights the top of the ranking, where the positives have to be found; ROC-AUC
weights all thresholds equally. When two metrics disagree on the ordering of
methods this close together, the honest conclusion is that they cannot be ranked
with this much data.

Raw numbers: [results/test_table.csv](results/test_table.csv), produced by
[test_table.py](test_table.py).

## Confusion tables

Threshold 0.5. Second operating point is the F1-optimal threshold picked on
validation (0.7015) and applied unchanged to test.

**Test, full (n=2397)**

| | pred. not produced | pred. produced |
|---|---:|---:|
| actual not produced | 1832 | 227 |
| actual produced | 62 | **276** |

| Metric @0.5 | Val | Test |
|---|---:|---:|
| accuracy | 0.885 | 0.879 |
| balanced accuracy | 0.843 | 0.853 |
| precision | 0.567 | 0.549 |
| recall | 0.785 | 0.817 |
| F1 | 0.658 | 0.656 |
| MCC | 0.603 | 0.604 |

At threshold 0.7015 the test table becomes 1931 / 128 / 99 / **239** →
precision 0.651, recall 0.707, F1 0.678.

### `symbolic` on test

Test AP / ROC-AUC are in the [final test table](#final-test-table--all-four-methods)
above. Its own decision rule is a score ≥ 0, which is a stricter operating point
than our threshold 0.5 because it is trained at natural prevalence with no class
weighting:

| Threshold | TN | FP | FN | TP | precision | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (its default) | 1988 | 71 | 138 | 200 | 0.738 | 0.592 | 0.657 | 0.913 |
| −0.617 (val F1-optimal) | 1923 | 136 | 100 | 238 | 0.636 | 0.704 | 0.669 | 0.902 |

## Balanced 50/50 subsets

`splits/val_equal.csv` (678 rows) and `splits/test_equal.csv` (676 rows): all
positives plus an equal number of seeded random negatives.

**Test-equal (n=676)**

| | pred. not produced | pred. produced |
|---|---:|---:|
| actual not produced | 292 | 46 |
| actual produced | 62 | **276** |

| Metric @0.5 | Val-equal | Test-equal |
|---|---:|---:|
| accuracy | 0.847 | 0.840 |
| precision | 0.896 | 0.857 |
| recall | 0.785 | 0.817 |
| F1 | 0.837 | 0.836 |
| AP | 0.921 | 0.913 |
| ROC-AUC | 0.924 | 0.918 |

Only negatives were removed, so the positive row is identical to the full test
table. Recall and ROC-AUC barely move (prevalence-independent); precision, AP and
accuracy jump (prevalence-dependent) — test precision 0.549 → 0.857 purely because
positives are 3.5× more common here than in reality.

Use the balanced sets to make accuracy interpretable (chance 50 % not 86 %). Use
the **full test** numbers for real-world expectations.

## Caveats

- **The embedding is not shown to beat a parts-list baseline, a
  parts-list-plus-order baseline, or a hand-engineered feature classifier** — on
  validation *or* on test (see above). This is the most important limitation of
  the result.
- **The split is optimistic.** All constructs are built from the same 1,384-module
  pool, so train and test share modules. This measures interpolation within a
  known pool, not generalisation to new chemistry. A **module-disjoint split** is
  the most valuable next experiment — and it is exactly the setting where the
  parts-list baselines should collapse and a sequence model could pull ahead.
- Only 339 validation positives, hence CIs of roughly ±0.05 AP.
- The `GGCGGC` junction linker is an assumption (~0.14 % of each sequence); the
  true cloning scar is not in the spreadsheets.
- 5 constructs (0.03 %) exceed the 10,240-token context and were truncated.
- The train+val refit hits `max_iter=3000` with a `ConvergenceWarning`. Checked:
  allowing 20,000 iterations converges at 3,077 and gives AP 0.7331 vs 0.7335 —
  one sample moves.

## Reproduce

```bash
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
cd /e/project1/scifi/thoeppe/genomeoceanbgc

python experiments/build_nt_dataset.py            # spreadsheets -> construct DNA
python experiments/prepare_data.py                # dedup + stratified split
sbatch experiments/embed.sbatch                   # 4 GPUs, ~9 min
sbatch experiments/sweep.sbatch                   # 190 configs, ~7 min
python experiments/analyze_val.py                 # bootstrap CIs
python experiments/confusion_and_balanced.py --test
sbatch experiments/symbolic.sbatch                # contributed method, tuning sweep
python experiments/run_symbolic.py --mode final --n-kmers 150000 --l2 3.0 --max-iter 2500
python experiments/test_table.py                  # final 4-method test table, ~12 min
```

Seeded (`SEED = 20260805`). Embedding 15,985 constructs took 9 min on 4× GH200
(~16.5 seq/s per GPU); the full sweep took 7 min on CPU.
