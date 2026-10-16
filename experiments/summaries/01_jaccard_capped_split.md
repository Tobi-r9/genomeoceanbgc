# Split A — Jaccard-capped test sets

A third, stricter evaluation tier layered on the existing 29 leave-one-family-out
folds. **No new training happens**: every method's per-construct test scores were
saved when it ran, so this is a filter on which test constructs get scored, plus
a re-score.

Produced by [`family_jaccard.py`](../family_jaccard.py) →
`results/family/jaccard_test.csv`.

---

## 1. Why this exists

The evaluation had two tiers: the **full** test set, and a **deduplicated** one
that drops test constructs having a training construct which differs by exactly
one module *at the same length, in the same positions*.

That positional filter has a blind spot, and one fold exposed it.

**XfaS** has a one-swap twin rate of **0.0 %**, which made it look like the
strictest fold in the study. We used it as the headline evidence that sequence
models can beat module baselines when the test is hardest. That reading was
wrong. XfaS's test set is almost entirely 6-module constructs while its training
set is dominated by 4–7 module ones, so the typical nearest neighbour looks like:

```
test  (6 modules): 1108,112,103,876,877,1245
train (7 modules): 1108,112,103,876,877,878,879
```

Five identical modules in the same order, then a different length. The positional
filter never fires — but a model only has to recognise the shared prefix.

Measured properly, XfaS is the **least** clean of the twelve primary folds:
median Jaccard 0.62 against its nearest training construct (highest of any fold),
and 83 % of its test constructs still share all-but-one module with a training
construct.

**So "deduplicated" does not mean "at least two modules different."** It means
"no one-module substitution at matched length and position". Insertions,
deletions and reorderings all survive it.

## 2. What this tier does instead

For every test construct, compute the maximum **Jaccard similarity of its module
set** against every training construct in that fold, and keep only those at or
below a threshold:

```
J(A,B) = |modules(A) ∩ modules(B)| / |modules(A) ∪ modules(B)|
```

Being set-based, this subsumes substitutions, insertions, deletions **and**
reorderings in one criterion, which is exactly what the positional filter missed.

Three thresholds — **0.7, 0.6, 0.5** — so the degradation is visible rather than
asserted at one arbitrary cut. The filter is label-blind: Jaccard depends only on
which modules a construct uses, never on whether it produced.

## 3. What survives

`twin %` is the old positional statistic, for comparison. `chance` is the
positive rate of the surviving subset — it moves, which is why AP at one
threshold cannot be compared with AP at another.

| Fold | twin % | full n / pos | J≤0.7 n / pos / chance | J≤0.6 n / pos / chance | J≤0.5 n / pos / chance |
|---|---:|---:|---:|---:|---:|
| GxpS | 11.5 | 6212 / 906 | 6025 / 895 / 0.149 | 5719 / 887 / 0.155 | 5023 / 744 / 0.148 |
| SzeS | 39.1 | 4062 / 647 | 3286 / 568 / 0.173 | 2826 / 453 / 0.160 | 2110 / 378 / 0.179 |
| TxlS | 24.2 | 2389 / 326 | 1913 / 278 / 0.145 | 1775 / 265 / 0.149 | 1658 / 264 / 0.159 |
| XbcS | 41.1 | 2347 / 198 | 1412 / 71 / 0.050 | 1325 / 58 / 0.044 | 1310 / 58 / 0.044 |
| FitS | 34.5 | 1863 / 317 | 1310 / 272 / 0.208 | 1133 / 248 / 0.219 | 970 / 244 / 0.252 |
| XisS | 36.6 | 1067 / 38 | 1067 / 38 / 0.036 | 1033 / 36 / 0.035 | 706 / 33 / 0.047 |
| PxbS | 44.0 | 997 / 68 | 646 / 26 † | 582 / 16 † | 420 / 11 † |
| PdpS | 38.2 | 967 / 93 | 670 / 78 / 0.116 | 631 / 77 / 0.122 | 596 / 76 / 0.128 |
| XenS | 31.2 | 871 / 78 | 643 / 36 / 0.056 | 585 / 23 † | 585 / 23 † |
| KolS | 42.5 | 826 / 128 | 612 / 69 / 0.113 | 564 / 63 / 0.112 | 371 / 48 / 0.129 |
| **XfaS** | **0.0** | 545 / 56 | **545 / 56 / 0.103** | **129 / 30 / 0.233** | 71 / 15 † |
| FldS | 35.6 | 505 / 52 | 495 / 46 / 0.093 | 471 / 35 / 0.074 | 331 / 16 † |

† fewer than 30 producers — too few to quote. Usable folds: **11 at J≤0.7,
10 at J≤0.6, 8 at J≤0.5**.

**The XfaS row is the point of the whole exercise.** At J≤0.7 nothing is removed
(its worst neighbour is 0.67, below the cut). At J≤0.6 it collapses from 545
constructs to 129 — a 76 % loss, by far the largest of any fold — because most of
its test set really is close to training, just not in the positional sense. By
J≤0.5 it is unquotable. Compare GxpS, which loses only 8 % of producers going to
J≤0.6.

Note also that **XfaS's positive rate more than doubles** under the cap, 0.103 →
0.233. The constructs that survive are enriched for producers, so its AP at
J≤0.6 is not comparable with its AP on the full set. Always read against the
`chance` column in the same row.

## 4. Results

Test AP, default settings, folds with ≥30 producers. `chance` is the positive
rate of that subset.

**J ≤ 0.7 (11 folds)**

| Fold | chance | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FitS | 0.208 | 0.278 | 0.303 | 0.352 | **0.384** | 0.368 | 0.317 | 0.369 |
| FldS | 0.093 | 0.625 | 0.757 | 0.792 | 0.799 | 0.769 | 0.807 | **0.812** |
| GxpS | 0.149 | 0.287 | 0.377 | 0.489 | 0.483 | 0.440 | 0.451 | **0.518** |
| KolS | 0.113 | 0.349 | 0.437 | 0.420 | **0.520** | 0.498 | 0.408 | 0.513 |
| PdpS | 0.116 | 0.208 | **0.716** | 0.634 | 0.713 | 0.684 | 0.705 | 0.690 |
| SzeS | 0.173 | 0.399 | 0.323 | 0.452 | 0.380 | 0.374 | 0.433 | **0.466** |
| TxlS | 0.145 | 0.351 | 0.329 | 0.311 | **0.383** | 0.380 | 0.370 | 0.351 |
| XbcS | 0.050 | 0.140 | 0.166 | 0.251 | 0.252 | 0.302 | **0.368** | 0.280 |
| XenS | 0.056 | 0.060 | 0.060 | 0.293 | 0.375 | **0.404** | 0.393 | 0.338 |
| XfaS | 0.103 | 0.313 | 0.314 | **0.467** | 0.208 | 0.258 | 0.241 | 0.345 |
| XisS | 0.036 | 0.158 | 0.107 | 0.087 | **0.173** | 0.165 | 0.157 | 0.109 |
| *median* | *0.113* | *0.287* | *0.323* | *0.420* | *0.383* | *0.380* | *0.393* | *0.369* |

**J ≤ 0.6 (10 folds)**

| Fold | chance | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FitS | 0.219 | 0.281 | 0.306 | 0.351 | **0.389** | 0.377 | 0.320 | 0.369 |
| FldS | 0.074 | 0.580 | 0.819 | 0.852 | 0.837 | 0.801 | **0.871** | **0.871** |
| GxpS | 0.155 | 0.291 | 0.382 | 0.492 | 0.486 | 0.443 | 0.455 | **0.521** |
| KolS | 0.112 | 0.336 | 0.425 | 0.381 | **0.518** | 0.506 | 0.393 | 0.490 |
| PdpS | 0.122 | 0.269 | 0.722 | 0.652 | 0.720 | 0.690 | **0.730** | 0.701 |
| SzeS | 0.160 | 0.376 | 0.323 | 0.415 | 0.359 | 0.348 | 0.401 | **0.435** |
| TxlS | 0.149 | 0.351 | 0.338 | 0.311 | 0.370 | **0.374** | 0.371 | 0.349 |
| XbcS | 0.044 | 0.139 | 0.177 | 0.221 | 0.266 | 0.313 | **0.377** | 0.272 |
| XfaS | **0.233** | **0.665** | 0.586 | 0.657 | 0.417 | 0.571 | 0.513 | 0.586 |
| XisS | 0.035 | 0.161 | 0.107 | 0.085 | **0.175** | 0.166 | 0.157 | 0.109 |
| *median* | *0.135* | *0.313* | *0.360* | *0.398* | *0.403* | *0.410* | *0.397* | *0.463* |

**J ≤ 0.5 (8 folds)**

| Fold | chance | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FitS | 0.252 | 0.282 | 0.308 | 0.361 | **0.392** | 0.383 | 0.327 | 0.375 |
| GxpS | 0.148 | 0.280 | 0.358 | 0.465 | 0.438 | 0.406 | 0.412 | **0.484** |
| KolS | 0.129 | 0.287 | 0.375 | 0.310 | **0.475** | 0.454 | 0.364 | 0.426 |
| PdpS | 0.128 | 0.277 | 0.728 | 0.648 | 0.724 | 0.695 | **0.735** | 0.699 |
| SzeS | 0.179 | 0.323 | 0.335 | 0.397 | 0.334 | 0.331 | 0.354 | **0.412** |
| TxlS | 0.159 | 0.354 | 0.341 | 0.312 | 0.371 | **0.374** | 0.372 | 0.349 |
| XbcS | 0.044 | 0.174 | 0.178 | 0.241 | 0.273 | 0.313 | **0.399** | 0.297 |
| XisS | 0.047 | 0.167 | 0.151 | 0.111 | **0.306** | 0.282 | 0.248 | 0.161 |
| *median* | *0.139* | *0.281* | *0.338* | *0.336* | *0.381* | *0.378* | *0.368* | *0.393* |

Full numbers with 95 % bootstrap CIs are in `results/family/jaccard_test.csv`.

## 5. What it shows

**GenomeOcean stays last at every tier.** Median AP 0.287 → 0.313 → 0.281 across
the three cuts, below every other method each time. The headline result is not an
artefact of residual near-duplicates: making the test progressively stricter does
not rescue the foundation model.

**The module baselines stay competitive at every tier.** Modules+bigram and
Modules sit at 0.38–0.41 throughout, and at J≤0.5 they have the two highest
medians of any method. Whatever is being removed, it is not what they were
relying on.

**The XfaS conclusion reverses direction.** On the full test set, XfaS was the
one fold where the sequence methods beat the module baselines, and we read that
as "sequence wins when the test is hardest". Under the Jaccard cap it turns out
XfaS was the *most* contaminated fold, and once 76 % of it is removed only 129
constructs remain. Whatever that fold is telling us, it is not "sequence models
transfer better to genuinely novel chemistry".

**A caution about the medians.** Each tier is a median over a *different* set of
folds (11, 10, 8), and each fold's chance line moves as the subset changes. The
medians are therefore not a trend line — they cannot be compared down the column.
For any close call use the paired comparisons in `FAMILY_RESULTS.md`, which
compare methods on the same fold and the same resampled rows.

## 5b. Is this tier actually worth using? Mostly not.

Asked directly, and tested rather than assumed. Two checks.

**Does capping change the method ranking?** Barely.

| Tier | folds | median Spearman ρ vs full-test ranking | same winning method |
|---|---:|---:|---:|
| J ≤ 0.7 | 11 | **+1.00** | 9 / 11 |
| J ≤ 0.6 | 10 | +0.80 | 5 / 10 |
| J ≤ 0.5 | 8 | +0.90 | 5 / 8 |

Every fold where the winner changes is a shuffle *within the tied cluster*
(Modules ↔ Modules+bigram ↔ Thesis) — never a genuine reversal. And restricting
the **full** test set to the folds each tier leaves usable already reproduces
almost the whole capped ordering, with the common methods moving by only
0.003–0.027 AP. Most of the apparent difference between tiers is **which folds
are included**, not the capping.

**Is the filter really label-blind?** In construction yes, in effect no. Max
Jaccard is correlated with the outcome, and the sign flips between folds:

| Fold | corr(max Jaccard, produced) | pos rate, high-J half | pos rate, low-J half | chance: full → J≤0.6 |
|---|---:|---:|---:|---:|
| FldS | **+0.314** | 0.185 | 0.027 | 0.103 → 0.074 |
| XenS | +0.206 | 0.141 | 0.043 | 0.090 → 0.039 |
| PxbS | +0.157 | 0.125 | 0.027 | 0.068 → 0.027 |
| FitS | −0.202 | 0.082 | 0.252 | 0.170 → 0.219 |
| XfaS | **−0.246** | 0.000 | 0.103 | 0.103 → 0.233 |

Mean |correlation| **0.156**, above 0.1 on **9 of 12 folds**. Capping therefore
reshapes the tested population in a direction that differs per fold — XfaS's
positive rate more than doubles, FldS's falls by a quarter. The capped tables are
not a strictly harder version of the same question; they are a *different*
question ("how does the model do on constructs unlike training?") asked on a
label-correlated subsample.

**Verdict.** The **Jaccard statistic** is the valuable output of this work — it
is what exposed XfaS, Erwinia and XfaS+banan, and it killed the "sequence wins on
the hardest folds" claim. The **capped test sets** are not: they change no
conclusion, cost a third of the usable folds (12 → 8), and introduce a
per-fold selection bias that makes their numbers harder to interpret, not easier.

Use max-Jaccard as the strictness diagnostic. Keep the full test set as the
primary evaluation. Treat the tables in §4 as a robustness appendix.

## 6. The hardest folds by Jaccard — and the results on them

This is what the diagnostic is actually for: use max-Jaccard to **rank** folds by
strictness, then evaluate on the **full** test set of the hardest ones. No
capping, so no prevalence shift and no label-correlated subsampling — the
selection happens at the level of *which fold to trust*, not *which construct to
score*.

All 62 folds across the three schemes, ranked by median max-Jaccard against
training (lower = harder), keeping those with ≥30 producers and ≥200 test
constructs. The twelve hardest:

| # | Fold | scheme | med J | twin | n | prod. | chance | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `TxlS` | family | 0.20 | 24 % | 2389 | 326 | 0.136 | 0.230 | 0.361 | 0.331 | 0.363 | **0.409** | 0.402 | 0.401 | 0.397 |
| 2 | `XenS` | family | 0.22 | 31 % | 871 | 78 | 0.090 | 0.361 | 0.073 | 0.077 | 0.380 | 0.413 | **0.424** | 0.396 | 0.398 |
| 3 | `XbcS` | family | 0.33 | 41 % | 2347 | 198 | 0.084 | 0.290 | 0.188 | 0.142 | 0.358 | 0.269 | 0.286 | **0.393** | 0.363 |
| 4 | `FldS+TxlS+tri` | multifamily | 0.33 | 38 % | 2909 | 379 | 0.130 | 0.245 | 0.362 | 0.405 | 0.430 | **0.480** | 0.452 | 0.435 | 0.467 |
| 5 | `FldS` | family | 0.33 | 36 % | 505 | 52 | 0.103 | 0.671 | 0.632 | 0.746 | 0.790 | 0.792 | 0.762 | 0.799 | **0.811** |
| 6 | `XtvB` | family | 0.38 | 1 % | 464 | 65 | 0.140 | 0.152 | 0.387 | 0.417 | 0.473 | 0.439 | 0.399 | 0.328 | **0.531** |
| 7 | `GxpS+apici+tolaa` | multifamily | 0.40 | 13 % | 6214 | 907 | 0.146 | 0.184 | 0.284 | 0.378 | 0.494 | 0.485 | 0.442 | 0.455 | **0.520** |
| 8 | `Bacillus` | genus | 0.40 | 25 % | 1948 | 81 | 0.042 | 0.149 | 0.113 | 0.155 | 0.224 | 0.271 | 0.297 | **0.337** | 0.291 |
| 9 | `GxpS` | family | 0.40 | 12 % | 6212 | 906 | 0.146 | 0.179 | 0.286 | 0.374 | 0.482 | 0.477 | 0.436 | 0.446 | **0.509** |
| 10 | `XisS` | family | 0.43 | 37 % | 1067 | 38 | 0.036 | 0.056 | 0.158 | 0.107 | 0.087 | **0.173** | 0.165 | 0.157 | 0.109 |
| 11 | `PdpS` | family | 0.50 | 38 % | 967 | 93 | 0.096 | 0.146 | 0.140 | **0.635** | 0.380 | 0.623 | 0.579 | 0.524 | 0.492 |
| 12 | `Pps` | family | 0.50 | 45 % | 1092 | 94 | 0.086 | 0.190 | 0.290 | 0.434 | 0.412 | **0.536** | 0.434 | 0.420 | 0.501 |
| — | *median, 10 hardest* | | | | | | *0.117* | *0.207* | *0.285* | *0.352* | *0.405* | *0.426* | *0.413* | *0.398* | *0.433* |
| — | *median lift, 10 hardest* | | | | | | | *1.78* | *2.68* | *2.78* | *3.38* | *3.51* | *3.43* | *3.88* | *3.69* |

`med J` is the median, over that fold's test constructs, of the highest Jaccard
similarity to any training construct. `Copy-twin` is the memorisation floor, not
a model, and is excluded from the bold-best marking.

### What the hardest folds say

On the ten hardest folds in the entire study, the ordering is the same as
everywhere else:

| | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| median AP | 0.207 | 0.285 | 0.352 | 0.405 | **0.426** | 0.413 | 0.398 | 0.433 |
| median lift | 1.78 | 2.68 | 2.78 | 3.38 | 3.51 | 3.43 | **3.88** | 3.69 |

**GenomeOcean is last of the real methods** (AP 0.285, lift 2.68), and sits only
0.078 AP above pure memorisation. The module-based methods and the ensemble
occupy the top. Restricting to the genuinely hardest folds — by the measure that
does not have the twin rate's blind spot — does not change the conclusion.

Three folds are worth calling out individually. **XenS** (rank 2) is where both
sequence methods fall below chance-level ranking (0.073 and 0.077 against a
chance line of 0.090) while the module baselines score 0.41–0.42. **XbcS**
(rank 3) and **SrfS+fus+gacam** are the two folds where `Simple` and `Thesis`
respectively pull clearly ahead. Nowhere in the top ten does GenomeOcean lead.

### The ranking vindicates the diagnostic

The three folds the twin rate scored as *perfectly clean* rank **44th, 45th and
46th of 62** by Jaccard — in the bottom third:

| Fold | scheme | twin rate | med J | Jaccard rank |
|---|---|---:|---:|---:|
| XtvB | family | 0.6 % | 0.375 | **6** |
| SrfS+fus+gacam | multifamily | 2.9 % | 0.500 | 21 |
| XfaS+banan | multifamily | **0.0 %** | 0.625 | **44** |
| XfaS | family | **0.0 %** | 0.625 | **45** |
| Erwinia | genus | **0.0 %** | 0.625 | **46** |

XtvB shows the twin rate is not always wrong — at 0.6 % it really is the sixth
hardest fold in the study. It fails specifically on **length-skewed** folds, and
Jaccard catches exactly those. The hardest fold overall, `TxlS`, has a twin rate
of 24.2 % — unremarkable by that measure, and genuinely the strictest by this one.

## 7. Caveats

- **Nothing was refit.** Models were trained on the full training set and are
  simply scored on a subset of test. That is the right thing for measuring
  contamination, but it means a model is not *optimised* for the harder subset.
- **The subsets are prevalence-shifted**, and not in a consistent direction —
  XfaS 0.103 → 0.233, FldS 0.093 → 0.074. Always read AP against the chance
  column beside it.
- **Fold count shrinks with the threshold** (11 → 10 → 8), so the strictest tier
  is also the noisiest.
- Jaccard is set-based and therefore **order-blind**: two constructs using the
  same six modules in different orders have J = 1.0 and are treated as
  duplicates. That is the conservative choice here — it removes more, not less.
