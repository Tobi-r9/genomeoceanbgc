# Split B — multi-family holdout (random triples)

Hold out **three biosynthetic families at once**, chosen at random, and test on
every construct that uses at least one of their modules.

Built by [`prepare_group_splits.py --mode multifamily`](../prepare_group_splits.py)
→ `splits/multifamily_splits.csv`. All methods re-run from scratch via
`--split-name multifamily`; results in `results/multifamily/`.

---

## 1. Why this split was proposed

The leave-one-family-out split has a known partial-ness. A construct is a chimera
drawing modules from 3–4 families, so holding out **one** family removes only
about **one of its six modules** — measured, a median of **0.229** of the average
test construct. The other five modules are still familiar, which is exactly why
the module baselines degrade so gracefully.

The intended fix: hold out three families simultaneously, so that a larger share
of each test construct is unfamiliar chemistry. Families were grouped **at random
rather than by relatedness**, deliberately — the [genus split](03_genus_split.md)
already covers biologically-nested grouping, and using related families for both
would make the two schemes redundant.

## 2. How the folds were built

The 116 usable family labels were shuffled with a fixed seed and partitioned into
39 groups of three. Eligibility is the same as the family split (test ≥200
constructs and ≥30 producers, train ≥5,000, test ≤45 % of the dataset), leaving
**26 folds, 10 of them primary** (twin rate ≤45 % and ≥500 test constructs).

| Group | train | test | pos rate | producers | twin rate |
|---|---:|---:|---:|---:|---:|
| GxpS+apici+tolaa | 8,255 | 6,214 | 0.146 | 907 | 13.0 % |
| Lgr+SypS+SzeS | 8,874 | 5,227 | 0.127 | 665 | 40.7 % |
| XbcS+XtpS+XtvS | 9,857 | 3,951 | 0.124 | 489 | 44.9 % |
| FldS+TxlS+tri | 10,005 | 2,909 | 0.130 | 379 | 37.5 % |
| PrtS+XgsC+XldS | 11,364 | 1,361 | 0.188 | 256 | 39.4 % |
| MchS+Pps+TblS | 12,653 | 1,094 | 0.088 | 96 | 44.1 % |
| JagS+PdpS+nosto | 11,484 | 977 | 0.097 | 95 | 24.6 % |
| AltS+XtvB+arthr | 12,810 | 581 | 0.124 | 72 | 20.3 % |
| XfaS+banan | 12,674 | 546 | 0.104 | 57 | **0.0 %** |
| AmbS+Cyclosporin+tyc | 12,471 | 533 | 0.146 | 78 | 40.3 % |

(the 10 primary folds; the other 16 are in `splits/multifamily_splits_summary.csv`)

## 3. **The scheme did not do what it was designed to do**

This is the main result, and it is a negative one about the design rather than
about any method.

**Unseen fraction of the average test construct:**

| Scheme | all folds | primary folds |
|---|---:|---:|
| family (hold out 1) | 0.229 | 0.259 |
| **multifamily (hold out 3)** | **0.243** | **0.239** |
| genus | 0.198 | 0.194 |

Holding out three families instead of one moved the unseen fraction from 0.229 to
0.243 — and on the primary folds it went *down*, 0.259 → 0.239.

The reason is arithmetic, and obvious in hindsight. The test set is defined as
constructs touching **any** of the three held-out families. That is a union, and a
union **widens** the test set rather than **deepening** it. Counting across all 26
folds:

| test constructs touching… | count | share |
|---|---:|---:|
| 1 of the 3 held-out families | 43,206 | **98.9 %** |
| 2 of the 3 | 469 | 1.1 % |
| all 3 | 0 | 0 % |

**99 % of test constructs still lose only one of their six modules.** Adding
families to the held-out set recruits *more constructs* into the test set, each
still missing about one module. It does not make any individual construct harder.

A second, smaller effect works against the design too: the ≤45 % test-size cap
means groups containing a large family are more likely to be rejected, so the
eligible groups skew toward combinations of small, rarely-used families.

**What would actually work** — and is the concrete recommendation from this
split:

1. **Require test constructs to use ≥2 held-out modules.** This selects for depth
   directly. From the table above there are only 469 such constructs across all
   26 folds, so it would need a much larger held-out set per fold to be viable.
2. **Group families that co-occur inside constructs**, rather than at random. If
   the three families are ones that habitually appear together, holding them out
   removes several modules from the same construct. This requires building the
   co-occurrence graph first and grouping on it.

Both change the eligibility arithmetic substantially and would need their own
feasibility pass.

## 4. Results

Test AP, default settings, the 10 primary folds. `chance` is that fold's positive
rate. `Copy-twin` is the memorisation floor, not a model.

| Group | twin | chance | n | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GxpS+apici+tolaa | 13 % | 0.146 | 6214 | 0.184 | 0.284 | 0.378 | 0.494 | 0.485 | 0.442 | 0.455 | **0.520** |
| Lgr+SypS+SzeS | 41 % | 0.127 | 5227 | 0.350 | 0.394 | 0.285 | 0.415 | 0.307 | 0.277 | 0.298 | **0.434** |
| XbcS+XtpS+XtvS | 45 % | 0.124 | 3951 | 0.295 | 0.433 | 0.278 | 0.431 | 0.337 | 0.351 | **0.440** | 0.424 |
| FldS+TxlS+tri | 38 % | 0.130 | 2909 | 0.245 | 0.362 | 0.405 | 0.430 | **0.480** | 0.452 | 0.435 | 0.467 |
| PrtS+XgsC+XldS | 39 % | 0.188 | 1361 | 0.312 | **0.633** | 0.460 | 0.587 | 0.407 | 0.471 | 0.517 | 0.566 |
| MchS+Pps+TblS | 44 % | 0.088 | 1094 | 0.197 | 0.335 | 0.433 | 0.427 | **0.514** | 0.404 | 0.440 | 0.512 |
| JagS+PdpS+nosto | 25 % | 0.097 | 977 | 0.116 | 0.156 | 0.627 | 0.419 | **0.638** | 0.592 | 0.607 | 0.513 |
| AltS+XtvB+arthr | 20 % | 0.124 | 581 | 0.186 | 0.422 | 0.400 | 0.400 | 0.424 | 0.442 | 0.308 | **0.486** |
| XfaS+banan | **0 %** | 0.104 | 546 | 0.104 | 0.335 | 0.338 | **0.459** | 0.244 | 0.284 | 0.270 | 0.363 |
| AmbS+Cyclosporin+tyc | 40 % | 0.146 | 533 | 0.339 | 0.469 | **0.547** | 0.313 | 0.472 | 0.454 | 0.538 | 0.349 |
| *median AP* | | *0.127* | | *0.221* | *0.378* | *0.403* | *0.429* | *0.448* | *0.442* | *0.440* | *0.477* |
| *median lift (primary)* | | | | *1.77* | *3.20* | *3.17* | *3.34* | *3.27* | *3.06* | *3.23* | *3.52* |
| *median lift (all 26)* | | | | *2.41* | *2.70* | *3.17* | *4.32* | *3.41* | *3.52* | *3.57* | *4.28* |

Full numbers with 95 % bootstrap CIs in `results/multifamily/all_methods_ci.csv`.

## 5. What the numbers show

**On the primary folds every method converges.** Median lift runs 3.06–3.52
across all seven methods — a spread of 0.46, against 1.5 on the family split
(2.34–3.83). **GenomeOcean is no longer clearly last here**: at 3.20 it sits
mid-pack, above Modules (3.06) and Symbolic (3.17).

This is the one place in the whole study where the embedding is not the weakest
method, so it deserves a clear caveat: it holds only on the 10 primary folds.
Across **all 26** folds GenomeOcean is last again (2.70, below the 3.17–4.32 of
everything else). The primary subset is chosen partly by twin rate, and the
convergence may be an artefact of which folds pass that filter rather than a
property of the split.

**The ensemble is the most consistent method**, top or near-top on 5 of 10
primary folds and highest median lift in both groupings (3.52 primary, 4.28 all).
That reproduces its behaviour on the family split: it does not raise the ceiling,
it avoids the floor.

**Fold-to-fold variance is enormous.** GenomeOcean ranges from 0.156
(JagS+PdpS+nosto) to 0.633 (PrtS+XgsC+XldS); Symbolic from 0.278 to 0.627. No
method is reliably best, and the per-fold ordering changes almost completely
between folds.

## 6. XfaS+banan — a third instance of the twin-rate artefact

`XfaS+banan` has a **0.0 % twin rate**, and on it the sequence methods beat the
module baselines (Thesis 0.459, Symbolic 0.338, GenomeOcean 0.335 against
Modules+bigram 0.244, Modules 0.284). This is the same pattern as XfaS on the
family split and Erwinia on the genus split.

It is the same artefact. Its length mismatch between test and training module
counts is **0.68**, against a median of 0.31 for all other folds. See
[`03_genus_split.md` §5](03_genus_split.md) for the full diagnosis and the
correlation across all 62 folds — the short version is that the one-swap twin
rate cannot see insertions or deletions, so a fold whose test constructs have a
different length distribution from training is scored as clean regardless of how
similar it really is.

Three independent partitions producing the same artefact on the same underlying
family is what makes this a measurement problem rather than a curiosity.

## 7. Caveats

- **The scheme failed at its stated goal** (§3). Its results are still a valid
  independent partition of the data, but it should not be described as "a harder
  split than leave-one-family-out" — by the unseen-fraction measure it is not.
- **10 primary folds of 26.** The non-primary folds have twin rates up to 82 %
  and should not carry conclusions.
- **The convergence of methods on primary folds** may be a selection effect of
  the primary filter rather than a property of the split; the all-26 medians tell
  a different story.
- Folds are not independent — all 26 train on overlapping subsets of the same
  15,985 constructs, and the same family appears in exactly one group, so groups
  are disjoint but their training sets overlap heavily.
- Random grouping means some groups mix families of wildly different sizes; the
  eligibility cap then filters those unevenly.
