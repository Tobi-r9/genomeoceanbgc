# The family split, explained from scratch

A guide to what we built and what we measured. It assumes no background: every
term is defined where it first appears. Nothing here replaces
[FAMILY_SPLIT.md](FAMILY_SPLIT.md) (how the split was built) or
[FAMILY_RESULTS.md](FAMILY_RESULTS.md) (the technical write-up) — this is the
version you can read start to finish.

---

## 1. What we are trying to predict

The lab builds engineered enzymes called NRPS constructs. Think of each one as a
**recipe**: a chain of about six parts clicked together in a specific order, like
Lego bricks. There is a kit of **1,173 different bricks** to choose from.

Each recipe was actually built and tested in the lab, and the result was written
down as one of two answers: it **produced** a compound, or it did not. We have
**15,985 recipes**, and only **14.1 % of them worked**.

The goal is to predict that answer from the design alone. If a model can tell in
advance which recipes will fail, the lab does not have to build them.

## 2. Where the bricks come from — and what a "family" is

The bricks were not invented. Each one was harvested from a real bacterium that
already had its own natural production line for making a compound. Biologists
call such a production line a **biosynthetic gene cluster**, or BGC.

Around **116 different natural production lines** contributed bricks to the kit.
Each has a name — `GxpS` (short for GameXPeptide synthetase), `SzeS`, `XbcS`,
`TxlS`, and so on.

> **This is what we mean by a "family": all the bricks that came out of the same
> natural production line.**

One important detail that drives everything later: **a recipe is a mixture**. A
typical construct uses bricks from three or four *different* families at once.
They are deliberately chimeric — that is the whole point of the engineering.

## 3. Why the old way of testing was too easy

To test any model you split the data: the model learns from one part and is
graded on another part it has never seen. The original setup shuffled all 15,985
recipes at random and dealt out 70 % to learn from, 15 % to tune on, and 15 % to
be graded on.

The problem is that every recipe is built from the same small kit of bricks. So
the graded recipes end up looking almost exactly like the learned ones. We
measured this:

| How similar is the test set to the training set? | |
|---|---:|
| Test recipes whose exact *set* of bricks also appears in training | 12.4 % |
| Test recipes with a training recipe that differs by **exactly one brick in one position** | **93.0 %** |

Ninety-three out of a hundred. That is like an exam where almost every question
is one you have already seen worked through, with one number changed.

**How much was that worth?** We measured it directly. Ignore modelling entirely
and just copy the label of the nearest one-brick twin in training: that alone
scores **AP 0.534** on the old test set. The best real model in the original
experiment scored 0.734. So roughly **three-quarters of the old headline number
was available from neighbour lookup**, with no learning at all.

**And how noisy is that trick?** Across the whole dataset there are 157,170 pairs
of recipes differing by exactly one brick in one position, and **18.3 % of them
have opposite labels**. So swapping a single brick flips the outcome about one
time in five. That is why copying the twin does not score even higher — and it is
a real statement about how sensitive these constructs are to a single change.

The consequence showed up in the original results: a trivial baseline that only
knows *which bricks are in the recipe* scored just as well as a 4-billion-
parameter foundation model. There was nothing difficult left to do.

## 4. The new split, step by step

### The rule

Pick one family — say GameXPeptide. Then:

1. Find **every brick** that came from that family.
2. **Delete all of those bricks from the training data.** Any recipe that uses
   even one of them is banned from training.
3. Those banned recipes become the **test set**.

Now the model has to make predictions about bricks it has never seen even once.

We repeated this once per family, giving **29 separate tests**. Each one is
called a **fold**.

### Why we remove bricks rather than recipes

The obvious alternative would be "test on recipes belonging to the GameXPeptide
family". But as noted in section 2, recipes are mixtures — almost none belongs to
a single family. Depending on the family, only **0 to 31 recipes** are built
purely from one. That is far too few to grade a model on. So a family has to be
defined by which *bricks* it owns.

### Four decisions that needed care

**Bricks with two parents.** Twenty-six bricks are recorded as coming from two
families at once, labelled things like `GxpS/XgsC`. We treat those as belonging
to *both*, so removing GameXPeptide removes them too. Otherwise a little bit of
GameXPeptide would sneak into training through the back door.

**Bricks with no family recorded.** Some bricks are marked only `mix` or
`unknown`. That is missing paperwork, not a real family, so we never hold those
out. They stay in training.

**The tuning set is also family-held-out.** Models have settings (called
hyperparameters) that must be chosen somewhere. If we chose them on a randomly
drawn slice, we would be tuning the model for the easy situation all over again.
So we hold out a few *additional, different* families to form the tuning set.
That gives three levels:

| Set | What it contains | What the model sees |
|---|---|---|
| **Training** | most recipes | never sees the tuning families or the test family |
| **Tuning** (validation) | recipes using bricks from a few other held-out families | used only to pick settings |
| **Test** | recipes using bricks from the one target family | graded once, at the very end |

**The split never looks at the answers.** Every decision above comes from the
brick catalogue spreadsheet. We never used the produced/not-produced column to
decide who goes where. This matters: if you use the answers to design the exam,
the exam is rigged, and any score you get is meaningless.

### Which families became folds

Of the 116 families, **29 qualify** as usable tests. A family is used only if its
test set has at least 200 recipes and at least 30 working ones (otherwise the
result is pure noise), and leaves at least 5,000 recipes to train on.

### Not every fold is equally strict

Removing a family cleanly does not guarantee a hard test. A test recipe uses ~6
bricks; if only one of them belonged to the removed family, the other five are
still familiar. So we re-measured the same "differs by one brick" statistic on
every new fold:

| | one-brick-twin rate |
|---|---:|
| old random split | 93.0 % |
| best new fold (XfaS) | **0.0 %** |
| GameXPeptide fold | 11.5 % |
| worst new fold (SypS) | 96.2 % |

We therefore sorted the folds into two groups:

- **12 primary folds** — genuinely strict (twin rate ≤ 45 %, at least 500 test recipes).
- **17 secondary folds** — kept for completeness, but still leaky. **Do not draw
  conclusions from these.**

## 5. How to read the two scores

Both scores below are about **ranking**. The model gives every test recipe a
score, we sort from most likely to least likely to work, and we ask how good that
ordering is.

### Average precision (AP)

Walk down the ranked list from the top and, at each working recipe you hit, ask
"of everything I have picked so far, what fraction actually works?" AP is the
average of those numbers. **1.0 is perfect.**

**The catch — and this is the single most important thing on this page.** A model
that ranks at random scores an AP equal to the fraction of recipes that work. So
the "chance line" is different for every fold: it ranges from **0.036** (only
3.6 % of that fold's recipes work) to **0.226**. An AP of 0.30 is excellent on a
fold where chance is 0.036 and poor on a fold where chance is 0.226.

**Always compare an AP against the chance-AP column in the same row.** Never
compare AP across rows.

### ROC-AUC

The probability that a randomly picked working recipe is ranked above a randomly
picked failing one.

- **1.0** = perfect
- **0.5** = coin flip, no information at all
- **below 0.5** = worse than guessing; the model has the ordering backwards

Unlike AP, its chance line is always 0.5, so ROC-AUC *is* comparable across
folds. It is less sensitive to what happens at the very top of the ranking, which
is why we report both.

### The [numbers in brackets] — confidence intervals

Each AP is measured on one particular set of test recipes. With a different set
of the same size you would get a somewhat different number. The bracket says how
much.

We estimate it by **bootstrapping**: draw a test set of the same size at random
*with replacement* from the one we have (some recipes appear twice, some not at
all), recompute AP, and repeat 2,000 times. The bracket is the range containing
the middle 95 % of those 2,000 values.

**How to use it:** if two methods' brackets overlap heavily, the data cannot tell
them apart, no matter which point estimate is higher. Folds with few working
recipes have very wide brackets — XisS has only 38 — so treat those rows gently.

## 6. The five methods, in plain language

Two of these know nothing about biological sequence at all. That turns out to
matter enormously.

### Modules (`modules`)

The simplest thing that could possibly work. The recipe becomes a row of 1,173
on/off switches — one per brick — with about six switched on. **That is the
entire input.** No sequence, no chemistry, not even the order of the bricks. It
asks: can you predict success purely from *which parts were used*?

### Modules + bigram (`modules+bigram`)

The same, plus one extra switch for every *adjacent pair* of bricks that occurs
("brick 105 followed by brick 1218"), which lets it notice that order matters.
Still no sequence.

### GenomeOcean (`emb_mean`)

The foundation model. GenomeOcean-4B is a 4-billion-parameter neural network
pre-trained on 11 million BGC DNA sequences. We feed it the construct's DNA and
take the 3,072 numbers it produces as a summary ("embedding"), then train a small
classifier on those numbers. **The big model itself is never retrained** — it is
used purely as a fixed feature extractor.

### Symbolic

A hand-designed classifier contributed by a collaborator, using no pretrained
model. It reads the protein sequence and computes: amino-acid composition, counts
of 12 known NRPS motifs, hashed counts of every 6-residue window, and the
identity of the first and last 50 residues. Then a logistic regression.

### The thesis method (`thesis`)

This is the pipeline from the master thesis in `NRPS-Models-and-data`
(`Recent_Models/model.py`), reimplemented here on our data.

**It uses no foundation model and no neural network at all.** Everything is
ordinary spreadsheet-style lookup and counting, fed to a decision-tree ensemble.
It is the most "classical" method in the comparison, and it is a **hybrid**: some
of its features are on/off switches like the module list, and some read the raw
sequence. That combination turns out to matter.

It builds four groups of features for each recipe.

**Group 1 — which organisms the bricks came from.** Look up each brick's source
bacterium in the donor catalogue, then create one on/off switch per organism
(about 100 of them). A switch is on if the recipe uses any brick from that
organism. This is the same *kind* of feature as the module list, just coarser:
instead of "which brick", it records "which bacterium the brick came from".

**Group 2 — do neighbouring bricks fit together?** Every brick has a recorded
"architecture", a short code for its internal domain layout such as `TCLAT`. The
catalogue also records what architecture each brick expects to see immediately
before and after it. For each junction in the chain we ask two yes/no questions:
does brick *i*'s expected downstream partner match brick *i+1*'s actual
architecture, and does brick *i+1*'s expected upstream partner match brick *i*'s?
The answers are then summed into four numbers: how many junctions matched in each
direction, and how many junctions there were. This is a measure of **chemical
plausibility**, and crucially it does not care which family the bricks came from.

**Group 3 — GC content.** For each brick's DNA, compute the percentage of bases
that are G or C. This is a basic property that affects how well a gene expresses.
Values are capped at 65 % so that a few extreme bricks cannot dominate, then the
recipe is summarised by the mean, median, maximum, and count.

**Group 4 — k-mer profiles of the sequence.** Take the full assembled DNA and the
full protein, and count every short substring of length 3, 4 and 5 — a "k-mer".
Weight them by TF-IDF, a standard text-processing trick where substrings that
appear in almost every recipe count for little and distinctive ones count for
more, then keep the 1,000 most useful of each. **This is the only group that
looks at the raw sequence.**

All four groups are concatenated into one long row of roughly 2,100 numbers, put
on a common scale, and handed to **XGBoost** — an ensemble of shallow decision
trees where each new tree is trained to correct the mistakes of the ones before
it. Their settings: 300 trees, maximum depth 6, learning rate 0.05, with the
rare "produced" class up-weighted so it is not ignored.

**Why it survives the family split better than the pure sequence methods.**
Groups 1, 2 and 3 are all robust to unfamiliar bricks, for two different reasons.
Group 1 is switches, so an organism seen only in the removed family has a switch
that was always off during training — the model simply has no opinion about it
and falls back on the rest. Groups 2 and 3 are summary statistics that stay on
the same scale no matter which bricks appear: "how many junctions are compatible"
and "what is the average GC" mean the same thing for a familiar recipe and a
completely novel one. Only group 4 has to extrapolate into unseen territory. So
roughly three-quarters of this model keeps working when a family is removed,
which is exactly what the results show.

**One caveat about our version.** In the original code the k-mer vocabulary and
the scaling were computed on the whole dataset before splitting, which lets
information from the test set leak into training. We fit them on the training
portion only. This makes our numbers slightly *lower* than theirs would be, but
honest.

### Simple (`simple`)

A minimal method we built after seeing the results above, to test a specific
idea. It is `modules+bigram` plus exactly **four numbers**: how many junctions in
the chain are architecturally compatible (counted in each direction), and the
median and maximum GC content.

The reasoning was that those four are **family-agnostic** — "how many junctions
fit" means the same thing for a familiar recipe and a completely novel one — so
they should transfer where k-mer features do not. We also deliberately left out
the thesis method's `upstream_len`, `downstream_len` and `GC_count`, which are
the same quantity (number of bricks minus one) computed three times and correlate
with the outcome at r = 0.001.

The result is covered in section 8; briefly, it matched the module baselines but
did not beat them, which was informative in a way we did not expect.

### Copy-twin (the memorisation floor, not a model)

This is not a method, it is a **yardstick**, and it appears in the tables so you
can see how much of any score is available for free.

For each test recipe, look in the training set for a recipe that differs by
exactly one brick in one position — a "twin". If there is one, predict whatever
label that twin had. If there are several, average them. If there is none,
predict the overall training rate.

No learning of any kind happens. So whatever this scores is the amount of
performance obtainable by pure memorisation, and a real method has to clear it
to be interesting. It is never marked as a "winner" in the tables for that
reason.

Two things to notice when reading it. On **XfaS**, the fold with no twins at all,
it lands exactly on chance (AP 0.103 against a chance line of 0.103, ROC-AUC
0.500) — which is the check that the split is doing what it claims. On **FldS**
it reaches 0.671 while the best real method gets 0.792, which tells you most of
that fold's impressive-looking scores are memorisation and should be discounted.

It does not appear in the deduplicated tables because there it would be chance by
construction: deduplication removes precisely the recipes that have twins.

### "Default settings" versus "retuned"

Every model has settings. We report two versions:

- **Default settings** — the settings chosen back on the easy random split,
  reused unchanged. All five methods have this.
- **Retuned per fold** — each fold picks its own best settings using its own
  family-held-out tuning set. Only three methods can do this; `symbolic` and
  `thesis` ship with fixed settings and no search procedure, so they appear only
  in the default tables.

**For a fair comparison across all methods, use the default-settings tables.**

**How much do the settings actually matter?** We measured this, and the answer
differs sharply by method. Across the 12 primary folds, using the old default
instead of that fold's best setting costs:

| Method | loss from using the old default | spread across all settings |
|---|---:|---:|
| GenomeOcean | **−0.165 AP** | 0.214 |
| Modules | −0.038 AP | 0.189 |
| Modules+bigram | −0.020 AP | 0.170 |

The nuance matters. The *total spread* across settings is similar for all three,
so it is not that GenomeOcean responds more wildly to its settings in general.
It is that **the best setting moves a long way when the split changes, and only
for GenomeOcean**. What was optimal on the easy random split is still nearly
optimal for the module baselines, but costs the embedding 0.165 AP.

Concretely, on the family folds GenomeOcean picks roughly **100x stronger
regularisation** than the random split chose (C ≤ 0.1 instead of C = 10, on 7 of
12 folds). That is a real form of brittleness: this model needs far heavier
regularisation once it has to generalise to new chemistry, and any comparison
that reuses a setting tuned on an easy split will understate it. Retuning is
worth +0.052 AP to GenomeOcean on test, and nothing at all to the module
baselines (−0.000 and −0.016) — for them, tuning on a small validation set just
adds noise.

---

## 7. Results

Every number below is measured **once**, on a test set the model never saw. Best
value in each row is bold. Remember: compare AP to the chance column in the same
row, never across rows.

Each table ends with a **median** row: the median of that column down the folds
shown. It is a convenient summary, but **do not use it to rank two methods that
are close.** The folds differ enormously in difficulty, so a median across them
reflects which folds sit in the middle as much as which method is better — we
were caught by exactly this when comparing the ensemble with Modules+bigram,
where the medians say one thing (0.398 vs 0.408) and a proper paired comparison
on the same folds says the opposite (+0.026 in the ensemble's favour). For any
close call, use the paired comparisons in section 8 rather than these medians.

**`twin rate`** is the share of that fold's test recipes that still have a
near-identical relative in training (differing by one brick). **Lower is a
stricter test.** Use it to weight how much you believe a row: a result on a
fold with a 90 % twin rate says little about transfer to new chemistry.

**PRIMARY — average precision (AP), default settings**

| Family | twin rate | test size | producers | chance AP | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GxpS | 12 % | 6212 | 906 | 0.146 | 0.179 [0.16–0.19] | 0.286 [0.26–0.31] | 0.374 [0.35–0.40] | **0.482** [0.45–0.52] | 0.477 [0.44–0.51] | 0.436 [0.40–0.47] | 0.446 [0.42–0.48] |
| SzeS | 39 % | 4062 | 647 | 0.159 | 0.359 [0.33–0.39] | 0.390 [0.35–0.43] | 0.336 [0.30–0.37] | **0.471** [0.43–0.51] | 0.407 [0.37–0.45] | 0.412 [0.38–0.45] | 0.450 [0.41–0.49] |
| TxlS | 24 % | 2389 | 326 | 0.136 | 0.230 [0.20–0.27] | 0.361 [0.31–0.41] | 0.331 [0.29–0.38] | 0.363 [0.31–0.42] | **0.409** [0.36–0.47] | 0.402 [0.35–0.46] | 0.401 [0.35–0.45] |
| XbcS | 41 % | 2347 | 198 | 0.084 | 0.290 [0.23–0.36] | 0.188 [0.16–0.24] | 0.142 [0.11–0.18] | 0.358 [0.30–0.43] | 0.269 [0.22–0.33] | 0.286 [0.24–0.35] | **0.393** [0.33–0.47] |
| FitS | 34 % | 1863 | 317 | 0.170 | 0.255 [0.22–0.29] | 0.271 [0.24–0.30] | 0.297 [0.26–0.34] | 0.356 [0.31–0.41] | **0.396** [0.34–0.45] | 0.376 [0.32–0.43] | 0.319 [0.28–0.37] |
| XisS | 37 % | 1067 | 38 | 0.036 | 0.056 [0.04–0.09] | 0.158 [0.10–0.24] | 0.107 [0.07–0.16] | 0.087 [0.06–0.15] | **0.173** [0.12–0.27] | 0.165 [0.11–0.27] | 0.157 [0.10–0.25] |
| PxbS | 44 % | 997 | 68 | 0.068 | 0.260 [0.17–0.38] | 0.144 [0.11–0.19] | 0.189 [0.14–0.25] | **0.289** [0.20–0.41] | 0.264 [0.18–0.38] | 0.283 [0.19–0.39] | 0.215 [0.15–0.31] |
| PdpS | 38 % | 967 | 93 | 0.096 | 0.146 [0.12–0.20] | 0.140 [0.10–0.20] | **0.635** [0.53–0.73] | 0.380 [0.30–0.48] | 0.623 [0.53–0.71] | 0.579 [0.48–0.69] | 0.524 [0.42–0.63] |
| XenS | 31 % | 871 | 78 | 0.090 | 0.361 [0.26–0.48] | 0.073 [0.05–0.11] | 0.077 [0.06–0.11] | 0.380 [0.29–0.50] | 0.413 [0.31–0.53] | **0.424** [0.32–0.55] | 0.396 [0.30–0.52] |
| KolS | 42 % | 826 | 128 | 0.155 | 0.485 [0.41–0.57] | 0.496 [0.41–0.58] | 0.549 [0.46–0.64] | 0.574 [0.49–0.66] | 0.588 [0.50–0.67] | **0.593** [0.50–0.69] | 0.499 [0.41–0.59] |
| XfaS | 0 % | 545 | 56 | 0.103 | 0.103 [0.08–0.13] | 0.313 [0.22–0.44] | 0.314 [0.22–0.43] | **0.467** [0.34–0.61] | 0.208 [0.15–0.30] | 0.258 [0.18–0.37] | 0.241 [0.17–0.34] |
| FldS | 36 % | 505 | 52 | 0.103 | 0.671 [0.54–0.78] | 0.632 [0.50–0.76] | 0.746 [0.64–0.84] | 0.790 [0.69–0.87] | 0.792 [0.69–0.87] | 0.762 [0.65–0.85] | **0.799** [0.70–0.88] |
| **median** |  |  |  | 0.103 | 0.258 | 0.278 | 0.322 | 0.380 | 0.408 | 0.407 | 0.398 |
| **median, cleanest folds (twin ≤ 35 %)** |  |  |  |  | 0.230 | 0.286 | 0.314 | 0.380 | 0.409 | 0.402 | 0.396 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  |  |  |  | 0.290 | 0.271 | 0.297 | 0.363 | 0.407 | 0.402 | 0.396 |

**PRIMARY — ROC-AUC, default settings**

| Family | twin rate | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|
| GxpS | 12 % | 0.550 | 0.762 | 0.832 | **0.839** | 0.816 | 0.801 | 0.818 |
| SzeS | 39 % | 0.690 | 0.771 | 0.719 | **0.813** | 0.792 | 0.797 | 0.806 |
| TxlS | 24 % | 0.640 | 0.782 | 0.767 | 0.788 | 0.811 | 0.807 | **0.828** |
| XbcS | 41 % | 0.756 | 0.750 | 0.606 | **0.862** | 0.810 | 0.811 | 0.853 |
| FitS | 34 % | 0.649 | 0.706 | 0.728 | 0.728 | **0.774** | 0.747 | 0.730 |
| XisS | 37 % | 0.675 | 0.863 | 0.839 | 0.784 | **0.898** | 0.885 | 0.879 |
| PxbS | 44 % | 0.712 | 0.760 | **0.820** | 0.799 | 0.794 | 0.815 | 0.788 |
| PdpS | 38 % | 0.646 | 0.604 | 0.907 | 0.882 | 0.927 | **0.930** | 0.891 |
| XenS | 31 % | 0.792 | 0.360 | 0.413 | 0.851 | 0.850 | **0.865** | 0.852 |
| KolS | 42 % | 0.775 | 0.826 | 0.878 | 0.841 | **0.889** | 0.877 | 0.844 |
| XfaS | 0 % | 0.500 | 0.751 | 0.790 | **0.800** | 0.730 | 0.762 | 0.754 |
| FldS | 36 % | 0.822 | 0.956 | 0.935 | **0.970** | 0.962 | 0.962 | 0.964 |
| **median** |  | 0.683 | 0.761 | 0.805 | 0.826 | 0.813 | 0.813 | 0.836 |
| **median, cleanest folds (twin ≤ 35 %)** |  | 0.640 | 0.751 | 0.767 | 0.800 | 0.811 | 0.801 | 0.818 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  | 0.690 | 0.750 | 0.719 | 0.813 | 0.810 | 0.807 | 0.828 |

### The same folds, with the near-duplicates removed

Every table above is scored on the full test set, which still contains recipes
that have a near-identical relative in training. The tables below drop those:
each fold keeps only test recipes with **no** one-module-substitution neighbour
anywhere in its training set, and everything is re-scored on what remains.

Nothing was refit — this is a filter of the evaluation set only, and it is
label-blind (having a twin depends on which modules a recipe uses, never on
whether it produced).

Two things change when you deduplicate, which is why the chance column matters
even more here. The test set shrinks, and its **positive rate moves** — on PdpS
it rises from 0.096 to 0.127, on KolS it falls from 0.155 to 0.118. So an AP here
is not comparable with the same method's AP in the tables above; compare each
number with the chance AP in its own row.

`kept` is the share of the original test set that survived. Folds marked † keep
fewer than 30 producers, which is too few to support any claim — they are shown
for completeness only.

**PRIMARY, DEDUPLICATED — average precision (AP), default settings**

| Family | orig twin rate | kept | test size | producers | chance AP | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GxpS | 12 % | 89 % | 5499 | 828 | 0.151 | 0.288 [0.26–0.31] | 0.370 [0.34–0.40] | **0.474** [0.44–0.51] | 0.465 [0.43–0.50] | 0.430 [0.40–0.47] | 0.434 [0.40–0.47] |
| SzeS | 39 % | 61 % | 2472 | 391 | 0.158 | 0.313 [0.27–0.36] | 0.304 [0.26–0.35] | **0.384** [0.34–0.43] | 0.329 [0.29–0.37] | 0.328 [0.29–0.37] | 0.344 [0.30–0.39] |
| TxlS | 24 % | 76 % | 1811 | 265 | 0.146 | 0.349 [0.30–0.41] | 0.335 [0.29–0.39] | 0.311 [0.27–0.37] | 0.370 [0.32–0.43] | **0.374** [0.32–0.43] | 0.371 [0.32–0.43] |
| XbcS | 41 % | 59 % | 1382 | 65 | 0.047 | 0.130 [0.10–0.19] | 0.164 [0.10–0.26] | 0.255 [0.18–0.37] | 0.240 [0.16–0.34] | 0.287 [0.20–0.41] | **0.354** [0.25–0.48] |
| FitS | 34 % | 65 % | 1220 | 261 | 0.214 | 0.278 [0.24–0.31] | 0.304 [0.27–0.35] | 0.352 [0.30–0.41] | **0.389** [0.34–0.45] | 0.377 [0.33–0.44] | 0.317 [0.27–0.37] |
| XisS | 37 % | 63 % | 676 | 32 | 0.047 | 0.166 [0.10–0.25] | 0.158 [0.11–0.23] | 0.110 [0.07–0.16] | **0.341** [0.22–0.52] | 0.313 [0.19–0.47] | 0.286 [0.18–0.45] |
| XenS † | 31 % | 69 % | 599 | 24 | 0.040 | 0.059 [0.02–0.18] | 0.061 [0.03–0.15] | 0.224 [0.11–0.42] | 0.315 [0.17–0.51] | 0.327 [0.18–0.52] | **0.366** [0.19–0.56] |
| PdpS | 38 % | 62 % | 598 | 76 | 0.127 | 0.277 [0.22–0.36] | 0.728 [0.63–0.81] | 0.648 [0.55–0.75] | 0.724 [0.63–0.81] | 0.695 [0.59–0.80] | **0.735** [0.64–0.82] |
| PxbS † | 44 % | 56 % | 558 | 22 | 0.039 | 0.069 [0.04–0.13] | 0.073 [0.05–0.12] | 0.137 [0.06–0.31] | 0.144 [0.06–0.31] | **0.168** [0.07–0.31] | 0.108 [0.05–0.21] |
| XfaS | 0 % | 100 % | 545 | 56 | 0.103 | 0.313 [0.22–0.44] | 0.314 [0.22–0.43] | **0.467** [0.34–0.61] | 0.208 [0.15–0.30] | 0.258 [0.18–0.37] | 0.241 [0.17–0.34] |
| KolS | 42 % | 58 % | 475 | 56 | 0.118 | 0.299 [0.20–0.42] | 0.420 [0.30–0.55] | 0.365 [0.26–0.51] | **0.490** [0.37–0.62] | 0.449 [0.33–0.60] | 0.369 [0.26–0.50] |
| FldS † | 36 % | 64 % | 325 | 14 | 0.043 | 0.667 [0.40–0.89] | 0.665 [0.40–0.88] | 0.621 [0.37–0.84] | 0.620 [0.38–0.85] | 0.621 [0.38–0.83] | **0.693** [0.46–0.90] |
| **median** |  |  |  |  | 0.110 | 0.283 | 0.309 | 0.358 | 0.356 | 0.351 | 0.360 |
| **median, cleanest folds (twin ≤ 35 %)** |  |  |  |  |  | 0.288 | 0.314 | 0.352 | 0.370 | 0.374 | 0.366 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  |  |  |  |  | 0.278 | 0.304 | 0.311 | 0.329 | 0.328 | 0.354 |

**PRIMARY, DEDUPLICATED — ROC-AUC, default settings**

| Family | orig twin rate | kept | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|
| GxpS | 12 % | 89 % | 0.753 | 0.820 | **0.832** | 0.803 | 0.788 | 0.805 |
| SzeS | 39 % | 61 % | 0.702 | 0.674 | **0.761** | 0.749 | 0.748 | 0.749 |
| TxlS | 24 % | 76 % | 0.758 | 0.756 | 0.734 | 0.769 | 0.771 | **0.793** |
| XbcS | 41 % | 59 % | 0.795 | 0.726 | 0.832 | 0.819 | 0.817 | **0.850** |
| FitS | 34 % | 65 % | 0.638 | 0.666 | 0.676 | **0.710** | 0.691 | 0.667 |
| XisS | 37 % | 63 % | 0.842 | 0.861 | 0.781 | **0.927** | 0.909 | 0.899 |
| XenS † | 31 % | 69 % | 0.392 | 0.535 | 0.816 | 0.848 | **0.886** | 0.844 |
| PdpS | 38 % | 62 % | 0.788 | **0.944** | 0.916 | 0.943 | 0.944 | 0.923 |
| PxbS † | 44 % | 56 % | 0.669 | 0.749 | 0.739 | 0.730 | **0.801** | 0.719 |
| XfaS | 0 % | 100 % | 0.751 | 0.790 | **0.800** | 0.730 | 0.762 | 0.754 |
| KolS | 42 % | 58 % | 0.743 | 0.859 | 0.788 | **0.879** | 0.872 | 0.809 |
| FldS † | 36 % | 64 % | 0.977 | 0.930 | 0.975 | 0.968 | 0.966 | **0.977** |
| **median** |  |  | 0.752 | 0.773 | 0.794 | 0.811 | 0.809 | 0.807 |
| **median, cleanest folds (twin ≤ 35 %)** |  |  | 0.751 | 0.756 | 0.800 | 0.769 | 0.771 | 0.793 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  |  | 0.702 | 0.674 | 0.761 | 0.769 | 0.771 | 0.793 |

**SECONDARY — average precision (AP), default settings**

| Family | twin rate | test size | producers | chance AP | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RirS | 60 % | 2174 | 260 | 0.120 | 0.232 [0.20–0.28] | 0.248 [0.21–0.29] | **0.556** [0.49–0.62] | 0.431 [0.37–0.50] | 0.492 [0.43–0.56] | 0.447 [0.39–0.51] | 0.433 [0.37–0.50] |
| XabS | 51 % | 2072 | 443 | 0.214 | 0.483 [0.44–0.53] | 0.470 [0.42–0.52] | 0.471 [0.43–0.52] | **0.641** [0.60–0.68] | 0.628 [0.58–0.67] | 0.617 [0.57–0.66] | 0.477 [0.43–0.52] |
| XtpS | 52 % | 1616 | 277 | 0.171 | 0.349 [0.30–0.40] | **0.569** [0.51–0.63] | 0.470 [0.41–0.53] | 0.435 [0.39–0.49] | 0.396 [0.35–0.45] | 0.398 [0.35–0.45] | 0.478 [0.42–0.54] |
| SldS | 49 % | 1381 | 196 | 0.142 | 0.509 [0.43–0.58] | 0.488 [0.42–0.57] | 0.595 [0.52–0.67] | **0.662** [0.59–0.73] | 0.468 [0.40–0.54] | 0.409 [0.34–0.48] | 0.519 [0.45–0.59] |
| SymS | 88 % | 1182 | 104 | 0.088 | 0.428 [0.34–0.53] | 0.232 [0.18–0.31] | 0.344 [0.27–0.44] | **0.538** [0.44–0.64] | 0.462 [0.37–0.56] | 0.448 [0.36–0.56] | 0.431 [0.34–0.54] |
| Pps | 45 % | 1092 | 94 | 0.086 | 0.190 [0.14–0.26] | 0.290 [0.22–0.39] | 0.434 [0.34–0.55] | 0.412 [0.32–0.52] | **0.536** [0.43–0.63] | 0.434 [0.33–0.54] | 0.420 [0.33–0.52] |
| XldS | 50 % | 1031 | 233 | 0.226 | 0.358 [0.31–0.42] | 0.600 [0.53–0.67] | **0.610** [0.54–0.68] | 0.609 [0.55–0.67] | 0.497 [0.44–0.57] | 0.584 [0.52–0.65] | 0.554 [0.49–0.63] |
| BacS | 60 % | 996 | 42 | 0.042 | 0.223 [0.13–0.38] | 0.125 [0.07–0.21] | 0.240 [0.15–0.37] | **0.400** [0.26–0.55] | 0.116 [0.08–0.17] | 0.156 [0.10–0.25] | 0.277 [0.16–0.42] |
| Bicornutin | 64 % | 940 | 174 | 0.185 | 0.403 [0.34–0.47] | 0.211 [0.18–0.25] | 0.551 [0.48–0.63] | **0.698** [0.62–0.77] | 0.617 [0.55–0.70] | 0.680 [0.61–0.76] | 0.661 [0.59–0.73] |
| CipS | 82 % | 865 | 49 | 0.057 | 0.341 [0.23–0.48] | 0.086 [0.05–0.16] | 0.229 [0.16–0.34] | 0.394 [0.28–0.52] | 0.427 [0.30–0.56] | 0.448 [0.32–0.58] | **0.469** [0.34–0.60] |
| Gra | 91 % | 592 | 30 | 0.051 | 0.315 [0.18–0.48] | 0.320 [0.16–0.49] | 0.180 [0.12–0.28] | **0.328** [0.20–0.51] | 0.299 [0.19–0.47] | 0.247 [0.15–0.41] | 0.286 [0.18–0.46] |
| CifS | 92 % | 512 | 90 | 0.176 | 0.635 [0.53–0.73] | 0.446 [0.36–0.56] | 0.541 [0.44–0.65] | 0.667 [0.57–0.75] | 0.644 [0.54–0.74] | 0.629 [0.52–0.72] | **0.727** [0.63–0.82] |
| SypS | 96 % | 494 | 44 | 0.089 | 0.398 [0.26–0.56] | 0.192 [0.11–0.31] | 0.159 [0.10–0.27] | 0.388 [0.27–0.54] | **0.526** [0.38–0.66] | 0.502 [0.35–0.65] | 0.456 [0.33–0.61] |
| AmbS | 53 % | 482 | 54 | 0.112 | 0.248 [0.17–0.37] | 0.360 [0.24–0.49] | 0.439 [0.32–0.57] | 0.381 [0.27–0.52] | 0.425 [0.32–0.57] | 0.403 [0.30–0.56] | **0.510** [0.39–0.65] |
| XtvB | 1 % | 464 | 65 | 0.140 | 0.152 [0.11–0.20] | 0.387 [0.28–0.51] | 0.417 [0.32–0.54] | **0.473** [0.36–0.59] | 0.439 [0.33–0.58] | 0.399 [0.30–0.53] | 0.328 [0.24–0.45] |
| BanS | 92 % | 355 | 37 | 0.104 | 0.496 [0.34–0.65] | 0.421 [0.28–0.60] | 0.387 [0.26–0.55] | **0.628** [0.47–0.77] | 0.611 [0.46–0.75] | 0.544 [0.39–0.70] | 0.590 [0.44–0.73] |
| AB1394 | 77 % | 320 | 34 | 0.106 | 0.308 [0.21–0.46] | 0.118 [0.08–0.16] | 0.340 [0.24–0.49] | **0.591** [0.42–0.73] | 0.319 [0.21–0.47] | 0.399 [0.28–0.58] | 0.318 [0.21–0.48] |
| **median** |  |  |  | 0.112 | 0.349 | 0.320 | 0.434 | 0.473 | 0.468 | 0.447 | 0.469 |

**SECONDARY — ROC-AUC, default settings**

| Family | twin rate | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|
| RirS | 60 % | 0.704 | 0.758 | **0.874** | 0.835 | 0.847 | 0.850 | 0.841 |
| XabS | 51 % | 0.762 | 0.758 | 0.830 | 0.852 | **0.856** | 0.848 | 0.807 |
| XtpS | 52 % | 0.738 | **0.898** | 0.836 | 0.861 | 0.810 | 0.824 | 0.854 |
| SldS | 49 % | 0.769 | 0.869 | **0.906** | 0.899 | 0.805 | 0.818 | 0.855 |
| SymS | 88 % | 0.856 | 0.804 | 0.816 | **0.885** | 0.874 | 0.867 | 0.864 |
| Pps | 45 % | 0.678 | 0.837 | 0.864 | 0.874 | **0.888** | 0.866 | 0.881 |
| XldS | 50 % | 0.685 | **0.854** | 0.828 | 0.833 | 0.795 | 0.834 | 0.837 |
| BacS | 60 % | 0.759 | 0.767 | 0.845 | 0.860 | 0.819 | 0.842 | **0.870** |
| Bicornutin | 64 % | 0.797 | 0.613 | 0.867 | **0.920** | 0.885 | 0.919 | 0.896 |
| CipS | 82 % | 0.896 | 0.581 | 0.869 | 0.907 | 0.902 | 0.904 | **0.918** |
| Gra | 91 % | 0.866 | 0.839 | 0.864 | **0.912** | 0.901 | 0.874 | 0.896 |
| CifS | 92 % | 0.864 | 0.817 | 0.853 | 0.872 | 0.864 | 0.857 | **0.920** |
| SypS | 96 % | 0.815 | 0.676 | 0.612 | 0.849 | 0.872 | 0.853 | **0.874** |
| AmbS | 53 % | 0.707 | 0.816 | 0.824 | 0.840 | 0.879 | 0.874 | **0.902** |
| XtvB | 1 % | 0.499 | 0.820 | **0.845** | 0.804 | 0.810 | 0.804 | 0.732 |
| BanS | 92 % | 0.851 | 0.768 | 0.830 | 0.880 | 0.895 | 0.866 | **0.907** |
| AB1394 | 77 % | 0.818 | 0.554 | 0.861 | **0.889** | 0.837 | 0.882 | 0.828 |
| **median** |  | 0.769 | 0.804 | 0.845 | 0.872 | 0.864 | 0.857 | 0.870 |

**PRIMARY — AP, retuned per fold**

| Family | twin rate | test size | producers | chance AP | GenomeOcean | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|
| GxpS | 12 % | 6212 | 906 | 0.146 | **0.489** [0.46–0.52] | 0.479 [0.45–0.51] | 0.437 [0.41–0.47] | 0.475 [0.44–0.51] |
| SzeS | 39 % | 4062 | 647 | 0.159 | 0.448 [0.41–0.49] | 0.407 [0.37–0.45] | 0.412 [0.38–0.45] | **0.453** [0.42–0.50] |
| TxlS | 24 % | 2389 | 326 | 0.136 | 0.378 [0.33–0.43] | 0.384 [0.33–0.44] | **0.402** [0.35–0.46] | 0.372 [0.32–0.43] |
| XbcS | 41 % | 2347 | 198 | 0.084 | 0.154 [0.12–0.20] | 0.159 [0.13–0.20] | 0.344 [0.29–0.41] | **0.389** [0.32–0.47] |
| FitS | 34 % | 1863 | 317 | 0.170 | 0.332 [0.29–0.38] | 0.331 [0.29–0.38] | **0.348** [0.30–0.40] | 0.347 [0.30–0.40] |
| XisS | 37 % | 1067 | 38 | 0.036 | 0.098 [0.06–0.16] | 0.153 [0.10–0.23] | **0.176** [0.11–0.29] | 0.135 [0.09–0.23] |
| PxbS | 44 % | 997 | 68 | 0.068 | 0.253 [0.18–0.35] | 0.264 [0.18–0.38] | **0.264** [0.18–0.38] | 0.258 [0.18–0.37] |
| PdpS | 38 % | 967 | 93 | 0.096 | 0.313 [0.24–0.41] | 0.600 [0.50–0.69] | 0.594 [0.49–0.70] | **0.612** [0.51–0.71] |
| XenS | 31 % | 871 | 78 | 0.090 | 0.103 [0.08–0.16] | **0.418** [0.31–0.53] | 0.346 [0.26–0.45] | 0.415 [0.31–0.54] |
| KolS | 42 % | 826 | 128 | 0.155 | 0.542 [0.45–0.63] | 0.436 [0.36–0.53] | **0.562** [0.47–0.66] | 0.445 [0.37–0.54] |
| XfaS | 0 % | 545 | 56 | 0.103 | **0.327** [0.22–0.46] | 0.197 [0.14–0.29] | 0.251 [0.16–0.37] | 0.229 [0.16–0.34] |
| FldS | 36 % | 505 | 52 | 0.103 | 0.701 [0.56–0.83] | 0.792 [0.69–0.87] | 0.780 [0.67–0.88] | **0.805** [0.71–0.88] |
| **median** |  |  |  | 0.103 | 0.329 | 0.395 | 0.375 | 0.402 |
| **median, cleanest folds (twin ≤ 35 %)** |  |  |  |  | 0.332 | 0.384 | 0.348 | 0.372 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  |  |  |  | 0.332 | 0.384 | 0.348 | 0.389 |

**PRIMARY — ROC-AUC, retuned per fold**

| Family | twin rate | GenomeOcean | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|
| GxpS | 12 % | **0.849** | 0.818 | 0.795 | 0.827 |
| SzeS | 39 % | 0.803 | 0.792 | 0.796 | **0.820** |
| TxlS | 24 % | 0.801 | 0.795 | **0.814** | 0.794 |
| XbcS | 41 % | 0.674 | 0.714 | **0.866** | 0.856 |
| FitS | 34 % | **0.761** | 0.706 | 0.722 | 0.728 |
| XisS | 37 % | 0.768 | 0.883 | **0.885** | 0.852 |
| PxbS | 44 % | **0.852** | 0.794 | 0.829 | 0.845 |
| PdpS | 38 % | 0.816 | 0.913 | **0.937** | 0.924 |
| XenS | 31 % | 0.522 | **0.862** | 0.860 | 0.852 |
| KolS | 42 % | 0.864 | 0.850 | **0.884** | 0.852 |
| XfaS | 0 % | **0.733** | 0.724 | 0.647 | 0.720 |
| FldS | 36 % | 0.951 | 0.963 | **0.972** | 0.965 |
| **median** |  | 0.802 | 0.807 | 0.844 | 0.849 |
| **median, cleanest folds (twin ≤ 35 %)** |  | 0.761 | 0.795 | 0.795 | 0.794 |
| **median, most-novel folds (≥ 33% of construct unseen)** |  | 0.761 | 0.792 | 0.814 | 0.820 |

**SECONDARY — AP, retuned per fold**

| Family | twin rate | test size | producers | chance AP | GenomeOcean | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|---|---|---|
| RirS | 60 % | 2174 | 260 | 0.120 | 0.411 [0.35–0.48] | **0.499** [0.44–0.56] | 0.427 [0.37–0.49] | 0.413 [0.36–0.48] |
| XabS | 51 % | 2072 | 443 | 0.214 | 0.543 [0.50–0.59] | 0.483 [0.44–0.53] | **0.601** [0.56–0.65] | 0.477 [0.43–0.52] |
| XtpS | 52 % | 1616 | 277 | 0.171 | 0.497 [0.44–0.56] | **0.525** [0.47–0.59] | 0.514 [0.46–0.58] | 0.521 [0.46–0.59] |
| SldS | 49 % | 1381 | 196 | 0.142 | **0.613** [0.54–0.68] | 0.468 [0.40–0.54] | 0.411 [0.35–0.48] | 0.546 [0.48–0.62] |
| SymS | 88 % | 1182 | 104 | 0.088 | 0.310 [0.24–0.40] | **0.462** [0.37–0.56] | 0.448 [0.36–0.56] | 0.427 [0.34–0.53] |
| Pps | 45 % | 1092 | 94 | 0.086 | 0.521 [0.41–0.63] | **0.536** [0.43–0.63] | 0.433 [0.34–0.54] | 0.477 [0.38–0.58] |
| XldS | 50 % | 1031 | 233 | 0.226 | **0.655** [0.59–0.71] | 0.497 [0.44–0.57] | 0.504 [0.44–0.58] | 0.554 [0.49–0.63] |
| BacS | 60 % | 996 | 42 | 0.042 | 0.171 [0.11–0.28] | 0.410 [0.27–0.57] | 0.211 [0.14–0.34] | **0.423** [0.28–0.57] |
| Bicornutin | 64 % | 940 | 174 | 0.185 | 0.567 [0.49–0.64] | 0.590 [0.52–0.68] | **0.640** [0.57–0.72] | 0.592 [0.52–0.68] |
| CipS | 82 % | 865 | 49 | 0.057 | 0.093 [0.07–0.13] | 0.323 [0.20–0.45] | 0.382 [0.26–0.52] | **0.488** [0.36–0.61] |
| Gra | 91 % | 592 | 30 | 0.051 | 0.258 [0.13–0.41] | **0.441** [0.27–0.61] | 0.247 [0.14–0.40] | 0.266 [0.17–0.43] |
| CifS | 92 % | 512 | 90 | 0.176 | 0.512 [0.41–0.61] | 0.654 [0.54–0.75] | 0.559 [0.45–0.66] | **0.664** [0.56–0.75] |
| SypS | 96 % | 494 | 44 | 0.089 | 0.349 [0.23–0.50] | **0.526** [0.38–0.66] | 0.503 [0.35–0.65] | 0.427 [0.31–0.59] |
| AmbS | 53 % | 482 | 54 | 0.112 | 0.328 [0.22–0.46] | 0.419 [0.30–0.57] | 0.423 [0.29–0.56] | **0.435** [0.31–0.58] |
| XtvB | 1 % | 464 | 65 | 0.140 | **0.414** [0.31–0.54] | 0.395 [0.29–0.53] | 0.335 [0.24–0.45] | 0.328 [0.24–0.45] |
| BanS | 92 % | 355 | 37 | 0.104 | 0.489 [0.34–0.64] | **0.602** [0.45–0.75] | 0.544 [0.39–0.70] | 0.473 [0.34–0.65] |
| AB1394 | 77 % | 320 | 34 | 0.106 | 0.440 [0.31–0.63] | 0.608 [0.44–0.77] | **0.697** [0.55–0.83] | 0.318 [0.21–0.48] |
| **median** |  |  |  | 0.112 | 0.440 | 0.497 | 0.448 | 0.473 |

**SECONDARY — ROC-AUC, retuned per fold**

| Family | twin rate | GenomeOcean | Modules+bigram | Modules | Simple |
|---|---|---|---|---|---|
| RirS | 60 % | 0.828 | 0.861 | **0.869** | 0.840 |
| XabS | 51 % | 0.820 | 0.812 | **0.848** | 0.807 |
| XtpS | 52 % | 0.827 | **0.861** | 0.829 | 0.855 |
| SldS | 49 % | **0.884** | 0.805 | 0.819 | 0.858 |
| SymS | 88 % | 0.830 | **0.874** | 0.867 | 0.869 |
| Pps | 45 % | 0.870 | **0.888** | 0.863 | 0.881 |
| XldS | 50 % | **0.856** | 0.795 | 0.799 | 0.837 |
| BacS | 60 % | 0.843 | **0.916** | 0.870 | 0.876 |
| Bicornutin | 64 % | 0.867 | 0.874 | **0.899** | 0.881 |
| CipS | 82 % | 0.675 | 0.853 | 0.899 | **0.923** |
| Gra | 91 % | 0.802 | 0.858 | 0.874 | **0.896** |
| CifS | 92 % | 0.812 | **0.873** | 0.827 | 0.854 |
| SypS | 96 % | 0.827 | 0.872 | 0.854 | **0.877** |
| AmbS | 53 % | 0.803 | 0.828 | 0.827 | **0.841** |
| XtvB | 1 % | **0.827** | 0.794 | 0.767 | 0.732 |
| BanS | 92 % | 0.823 | **0.884** | 0.866 | 0.872 |
| AB1394 | 77 % | 0.871 | 0.921 | **0.940** | 0.828 |
| **median** |  | 0.827 | 0.861 | 0.863 | 0.858 |

## 8. What the tables actually say

### a. Everything got much harder, and that was the point

On the old random split every method scored an AP of 0.69–0.73 against a chance
line of 0.141 — roughly a 5× lift. On the strict folds nothing comes close to
that consistently. This is not a bug; it is the easy exam being replaced by a
hard one. **The old numbers should not be quoted as evidence of family
transfer.**

### b. The module baselines did not collapse — and that is the headline

The prediction going in was that a family split would break the module-list
baselines (which just memorise which bricks work) and let the sequence model
pull ahead. Look at the "Modules" column: it wins outright on several folds
and is close on most others, despite knowing nothing about chemistry.

There is a mechanical reason. Holding out one family removes only about **one of
the six bricks** in a typical test recipe. The module-list model has a switch per
brick; a brick it never saw in training has a switch that was always off, so it
learns no weight for it and it contributes nothing at test time. The model
quietly falls back on the five bricks it *does* recognise. **It degrades
gracefully by ignoring the unknown.**

### c. The sequence-based methods cannot ignore anything

GenomeOcean and Symbolic read the actual sequence. The unfamiliar brick's
residues are physically in that sequence, so every feature they compute shifts
into territory the model has never seen. They cannot abstain, so they
extrapolate — and extrapolation can go backwards.

Look at the **XenS** row. Both sequence methods score an ROC-AUC **below 0.5**
(0.360 and 0.413), meaning they rank working recipes *worse* than random, while
both module baselines are unaffected (0.865 and 0.850). XenS is the fold where
the largest share of each test recipe — 46 % — was held out.

### d. The hybrid handles this best

The thesis method carries both kinds of feature, and it degrades least of all
five as more of the recipe becomes unfamiliar. Its organism switches can abstain
like a module list; its GC and compatibility features are summary statistics that
stay meaningful whatever bricks appear; only its k-mer features have to
extrapolate. On XenS it holds at AP 0.380 while the pure sequence methods fall
apart.

### e. On XfaS the ordering flips — but XfaS is NOT the strictest fold

On **XfaS** the sequence-reading methods win: the thesis method (0.467) and the
two sequence methods (0.313, 0.314) all beat both module baselines (0.258,
0.208). We originally read this as "the one fold where the original hypothesis
holds, and it is the strictest test we have". **That reading was wrong, and the
error is instructive.**

XfaS has a twin rate of 0.0 %, but the twin rate only counts training recipes
that differ by exactly one brick **at the same length, in the same positions**.
It is blind to a training recipe that differs by an insertion or deletion. XfaS
is full of exactly that case — its test set is almost entirely 6-brick recipes
while training is dominated by 4-to-7-brick ones, so the pattern below is
systematic:

```
test  (6 bricks): 1108,112,103,876,877,1245
train (7 bricks): 1108,112,103,876,877,878,879
```

Five identical bricks in the same order, then a different length. The twin filter
never fires, but a model only has to recognise the shared prefix.

Measured properly, XfaS is the **least** clean of the twelve primary folds, not
the cleanest: the median XfaS test recipe has Jaccard 0.62 against its nearest
training recipe — the highest of any fold — and **83 % of its test recipes still
share all-but-one brick (as a set) with some training recipe**.

So the flip on XfaS is not evidence that sequence models win when the test is
hardest. If anything it is the opposite: they win on the fold with the most
residual similarity. Section 7's Jaccard-capped tables are the version of this
question that can actually be trusted.

Treat XfaS gently in any case: it has just 56 working recipes.

### f. Read the brackets before believing any ranking

Many rows have heavily overlapping intervals. On KolS, for instance, four of the
five methods sit between 0.496 and 0.593 with brackets spanning ~0.2 — that row
does not rank anything. The rows with genuinely separated intervals are the ones
worth arguing about.

### g. The minimal method matches everything, once it is tuned

`simple` is `modules+bigram` plus four family-agnostic numbers — no k-mers, no
sequence, no pretrained model. Paired bootstrap against each rival on the 12
primary folds, on AP:

| `simple` (retuned) vs | better on | median ΔAP |
|---|---:|---:|
| Thesis | 6 / 12 | **+0.001** |
| Modules+bigram | 7 / 12 | +0.011 |
| Modules | 6 / 12 | +0.007 |

All three medians are indistinguishable from zero. **A model with four chemistry
numbers on top of the module list does as well as the thesis method's ~2,100
features, and as well as the 4-billion-parameter embedding pipeline.**

Two honest qualifications. First, this holds for the *retuned* comparison; with
the borrowed default setting `simple` sits slightly behind (median ΔAP −0.006 to
−0.014), because that default was chosen for the module baselines and never for
this feature set. Second, the four extra numbers are not what is doing the work:
retuned `simple` and retuned `modules+bigram` differ by +0.011 median AP, which
is nothing. **Essentially all the signal is in which modules are present.**

The useful negative result is therefore about the k-mers: dropping the thesis
method's entire sequence block costs nothing detectable on median. It still
matters per fold — the differences are individually significant on 4 of 12 folds,
in both directions — so the k-mers are high-variance rather than useless. They
help on some families and fail badly on the hardest ones.

### h. Removing the near-duplicates does not change the conclusions

On the 9 folds whose deduplicated subset stays large enough to quote, the
ordering is stable: GenomeOcean remains last, and Thesis, Modules+bigram and
Simple stay bunched at the top. So the headline is not an artefact of the
leftover near-duplicates.

Two details worth noting when reading those tables. Deduplication changes each
fold's positive rate in *both* directions (PdpS 0.096 → 0.127, KolS 0.155 →
0.118), which is why AP there cannot be compared with AP in the full-test tables.
And **XfaS is identical in both**, to three decimals, because it had no
near-duplicates to remove — a useful check that the filter does nothing when
there is nothing to do.

### i. Combining Thesis with Modules+bigram does help

The two are not variants of each other — they see genuinely different things:

| | Modules+bigram | Thesis |
|---|---|---|
| exact module identity | 1,173 switches | — |
| which module follows which | yes | — |
| organism identity | — | 100 switches (a ~12x coarsening: 1,011 of 1,173 modules share their organism with >= 5 others) |
| architecture / GC / k-mers | — | yes |

Neither is a superset of the other, and they win on different folds, so we mixed
them. Both bases keep their default settings; the only thing chosen is the mixing
weight, and it is chosen **once globally on validation across all folds** (not
per fold — see below), then applied unchanged to test. Validation preferred
w = 0.65, i.e. 65 % Thesis, on a smooth curve (mean val AP 0.387 at pure
Modules+bigram, 0.458 at the optimum, 0.440 at pure Thesis).

Paired bootstrap on test, 12 primary folds:

| ensemble vs | better on | median ΔAP | significantly ahead | significantly behind |
|---|---:|---:|---:|---:|
| Thesis | **11 / 12** | **+0.022** | 7 folds | 1 fold |
| Modules+bigram | 7 / 12 | **+0.026** | 4 folds | 2 folds |

**The ensemble is a real improvement over Thesis, and a modest one over
Modules+bigram.** What it mostly buys is the absence of disasters: each base has a
fold where it collapses relative to the other (Modules+bigram 0.208 vs Thesis
0.467 on XfaS; Thesis 0.380 vs Modules+bigram 0.623 on PdpS), and the blend is
never the worst option anywhere.

**A warning about how to measure this.** Comparing the *median AP across folds*
of each method suggests the opposite conclusion (ensemble 0.398 versus
Modules+bigram 0.408). That statistic is misleading here: the folds differ so much
in difficulty that a median across them is dominated by which folds happen to sit
in the middle, not by which method is better. The paired bootstrap compares the
two methods on the **same** fold and the same resampled rows, which is the
like-for-like question. Where the two disagree, trust the paired test.

**Per-fold weight selection does not work.** Choosing the weight separately on
each fold's validation set scored *worse* (median test AP 0.395) than a single
global weight, and worse than a plain 50/50 blend. The chosen weights ranged from
0.15 to 1.00 with no pattern — several of these validation sets have under 100
producers, so a 42-option search fits noise. An oracle weight chosen on test
would reach 0.441, so roughly +0.03 AP of headroom exists that neither selection
strategy can reach.

If this is worth pursuing, the promising direction is choosing the weight from a
**property of the fold** rather than from validation performance — the twin rate
and the fraction of the construct that is unseen both correlate with which base
wins. With 12 folds that meta-model would be badly underpowered, but it is the
shape of the thing that could work.

## 9. Honest limitations

- **The folds are not independent.** All 29 train on overlapping subsets of the
  same 15,985 recipes, so they are not 29 separate experiments and the spread
  across them understates the real uncertainty.
- **Some folds are tiny.** Test working-recipe counts range from 30 to 906. The
  small ones have intervals too wide to support any claim.
- **Residual leakage remains** on most primary folds. The per-fold twin rate
  ranges 0–44 %, but that statistic is positional and understates similarity:
  measured by Jaccard against the nearest training recipe, no primary fold is
  fully clean, and XfaS — which the twin rate scores as perfect — is the worst.
- **Two methods could not be retuned**, because their published code contains no
  tuning procedure. They appear only in the default-settings tables.
- **Training-set size varies** from 7,655 to 13,068 recipes across folds, so a
  fold that holds out a big family also penalises every method with less data.
- **The comparison partly rewards abstention rather than chemistry.** A model
  that can ignore unfamiliar parts has a structural advantage here that has
  nothing to do with understanding biosynthesis. That is worth remembering
  before concluding "the foundation model does not transfer".

## 10. Where everything lives

| File | What it is |
|---|---|
| [FAMILY_SPLIT.md](FAMILY_SPLIT.md) | how the 29 folds were built, and the checks on them |
| [FAMILY_RESULTS.md](FAMILY_RESULTS.md) | the technical results write-up, including issues found |
| [REPORT.md](REPORT.md) | the original random-split experiment, unchanged |
| `results/family/all_methods_ci.csv` | every number in section 7, machine-readable |
| `splits/family_splits.csv` | the fold assignments themselves |
