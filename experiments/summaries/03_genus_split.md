# Split C — leave-one-genus-out

Hold out every biosynthetic family belonging to one **bacterial genus**, and test
on every construct that uses at least one of its modules.

Built by [`prepare_group_splits.py --mode genus`](../prepare_group_splits.py) →
`splits/genus_splits.csv`. All methods re-run from scratch on the new folds via
`--split-name genus`; results in `results/genus/`.

---

## 1. Why this split

The original leave-one-family-out split holds out one biosynthetic family (a
`cluster` label such as GxpS or SzeS). A construct is a chimera of 3–4 families,
so holding out one removes only about **one of its six modules**.

Genus is the natural coarsening. Instead of "can it transfer to a different
synthetase", it asks **"can it transfer to a different organism"** — every family
that came out of, say, *Bacillus* is removed at once. It is also the split a
biologist would reach for first, because genus is a real taxonomic boundary
rather than an engineering label.

Everything else is inherited from the family split and unchanged: module-disjoint
holdout, hybrid labels counting for every parent, `mix`/`unknown` never held out,
validation itself held out by group, and the whole construction label-blind.

## 2. The folds

Of 41 genus labels, **7 leave enough data to be usable** (test ≥200 constructs
and ≥30 producers, train ≥5,000, test ≤45 % of the dataset).

**Xenorhabdus and Photorhabdus are excluded** — the two biggest contributors.
Holding out either would put 75 % and 47 % of the dataset into test respectively,
leaving too little to train on and making the comparison a sample-size study
rather than a transfer study. This is a real limitation of the scheme: the two
genera that matter most to this library cannot be tested this way.

| Genus | train | val | test | test pos rate | producers | twin rate | primary |
|---|---:|---:|---:|---:|---:|---:|:-:|
| Pseudomonas | 9,098 | 3,592 | 3,295 | 0.098 | 324 | 76.0 % | |
| Paenibacillus | 11,644 | 2,081 | 2,260 | 0.026 | 58 | 57.8 % | |
| Bacillus | 10,840 | 3,197 | 1,948 | 0.042 | 81 | 24.8 % | ✓ |
| Brevibacillus | 12,557 | 2,320 | 1,108 | 0.030 | 33 | 58.6 % | |
| Pedobacter | 12,752 | 2,266 | 967 | 0.096 | 93 | 34.5 % | ✓ |
| Erwinia | 11,572 | 3,869 | 544 | 0.101 | 55 | **0.0 %** | ✓ |
| Lautropia | 12,511 | 3,154 | 320 | 0.106 | 34 | 59.4 % | |

Only **3 of 7 are primary** (twin rate ≤45 % and ≥500 test constructs), a worse
proportion than the family split's 12 of 29. Holding out more at once does not
automatically produce cleaner folds — see §5.

Note how low the positive rates are: Paenibacillus 0.026, Brevibacillus 0.030,
Bacillus 0.042. These genera are mostly non-producers in this library, so their
folds have very few producers to rank.

## 3. Results

Test AP, default settings. `chance` is that fold's positive rate — always read AP
against it. `Copy-twin` is the memorisation floor, not a model.

| Genus | twin | chance | n | Copy-twin* | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pseudomonas | 76 % | 0.098 | 3295 | 0.260 | 0.173 | 0.158 | 0.376 | 0.416 | 0.391 | 0.345 | **0.424** |
| Paenibacillus | 58 % | 0.026 | 2260 | 0.094 | 0.029 | 0.127 | 0.176 | 0.139 | 0.121 | 0.196 | **0.200** |
| **Bacillus** | 25 % | 0.042 | 1948 | 0.149 | 0.113 | 0.155 | 0.224 | 0.271 | 0.297 | **0.337** | 0.291 |
| Brevibacillus | 59 % | 0.030 | 1108 | 0.294 | 0.241 | 0.149 | 0.352 | 0.230 | 0.179 | 0.338 | **0.362** |
| **Pedobacter** | 35 % | 0.096 | 967 | 0.142 | 0.140 | **0.636** | 0.391 | 0.625 | 0.579 | 0.567 | 0.512 |
| **Erwinia** | 0 % | 0.101 | 544 | 0.101 | 0.286 | 0.290 | **0.493** | 0.211 | 0.220 | 0.218 | 0.357 |
| Lautropia | 59 % | 0.106 | 320 | 0.356 | 0.118 | 0.337 | **0.496** | 0.329 | 0.399 | 0.268 | 0.481 |
| *median lift, all 7* | | | | *3.35* | *1.76* | *3.72* | *4.88* | *5.40* | *4.73* | *5.89* | *5.32* |

Bold = primary fold. Full numbers with 95 % bootstrap CIs in
`results/genus/all_methods_ci.csv`.

## 4. What it shows

**GenomeOcean collapses harder here than anywhere else.** Median lift over chance
is **1.76** across the seven folds — the worst of any method, below even the
memorisation floor (3.35). On Paenibacillus it scores AP 0.029 against a chance
line of 0.026: no signal at all. On Lautropia 0.118 against chance 0.106, again
essentially nothing. Whatever the frozen 4B embedding encodes, it does not survive
a change of organism.

**The module baselines and the thesis method hold up.** Modules+bigram 5.40,
Simple 5.89, Ensemble 5.32, Thesis 4.88. The ordering is broadly the same as the
family split, which is reassuring — this is a genuinely different partition of the
data and it reproduces the headline.

**Symbolic is wildly fold-dependent**: best of all methods on Pedobacter (0.636),
near-useless on Pseudomonas (0.158, below the memorisation floor of 0.260) and on
Brevibacillus (0.149). Consistent with the high-variance behaviour seen on the
family split.

**Copy-twin is high on the leaky folds** — 0.356 on Lautropia and 0.294 on
Brevibacillus, both above several real methods. Those two folds have twin rates
of 59 %, so their numbers should carry little weight.

## 5. The Erwinia trap — and a general lesson about the twin rate

Erwinia has a **0.0 % twin rate**, and on it the sequence-reading methods beat the
module baselines: Thesis 0.493, Symbolic 0.290, GenomeOcean 0.286, against
Modules+bigram 0.211 and Modules 0.220. Taken at face value this is the
"sequence models win when the test is genuinely hard" result.

**It is not.** Erwinia is the same artefact as XfaS in the family split:

| | XfaS (family) | Erwinia (genus) |
|---|---:|---:|
| twin rate | 0.0 % | 0.0 % |
| median Jaccard to nearest training construct | 0.62 | **0.62** |
| share all-but-one module with a training construct | 83 % | 68 % |
| test constructs with 6 modules | 500 / 545 | 500 / 544 |
| constructs surviving a J ≤ 0.6 cap | 129 (24 %) | 228 (42 %) |

Both folds have a test set that is almost entirely 6-module constructs while
training spans 4–7 modules. The one-swap twin filter only compares constructs of
**equal length in matched positions**, so a length-skewed fold mechanically has
almost no twins — regardless of how similar its constructs actually are.

Testing this across all 62 folds in the three schemes, twin rate is negatively
correlated with how much the test and training module-count distributions differ:

| Scheme | folds | corr(twin rate, length mismatch) |
|---|---:|---:|
| family | 29 | −0.50 |
| genus | 7 | −0.74 |
| multifamily | 26 | −0.41 |
| **all** | **62** | **−0.46** |

And the folds the twin rate calls perfectly clean are precisely the length-skewed
ones — median length mismatch **0.68** versus **0.31** for every other fold:

| Scheme | Fold | twin rate | length mismatch |
|---|---|---:|---:|
| family | XfaS | 0.0 % | 0.68 |
| genus | Erwinia | 0.0 % | 0.72 |
| multifamily | XfaS+banan | 0.0 % | 0.68 |
| family | XtvB | 0.6 % | 0.36 |
| multifamily | SrfS+fus+gacam | 2.9 % | 0.38 |

> **The one-swap twin rate systematically mislabels length-skewed folds as
> clean.** It is still a useful lower bound on contamination, but it must not be
> used on its own to rank folds by strictness. The Jaccard-based measure in
> [Split A](01_jaccard_capped_split.md) is set-based and does not have this blind
> spot.

This is the single most useful thing the genus split produced: it replicated the
XfaS anomaly on an independent partition, which is what turned a one-fold oddity
into a diagnosable measurement error.

## 5b. Erwinia under an actual similarity cap

Applying the Jaccard tiers from [Split A](01_jaccard_capped_split.md) to the genus
folds settles it. Erwinia's worst neighbour sits at J = 0.67, so **J ≤ 0.7 removes
nothing at all** — which is precisely why it looked clean. Tighten by one step and
it falls apart:

| Erwinia | n | producers | chance | Thesis | GenomeOcean | Symbolic | Modules+bigram |
|---|---:|---:|---:|---:|---:|---:|---:|
| J ≤ 0.7 | 544 | 55 | 0.101 | 0.493 | 0.286 | 0.290 | 0.211 |
| J ≤ 0.6 | 228 | 27 † | 0.118 | 0.480 | 0.292 | 0.283 | 0.168 |
| J ≤ 0.5 | 175 | 10 † | 0.057 | 0.139 | 0.117 | 0.180 | 0.094 |

† below 30 producers — not quotable.

**So the Erwinia "sequence wins" result exists only at the tier where nothing is
removed.** By J ≤ 0.6 the fold is already too small to support a claim, and by
J ≤ 0.5 every method has collapsed to near chance. It should not be cited as
evidence for sequence models.

Only **two genus folds survive a real similarity cap** with enough producers:

| Genus | chance | GenomeOcean | Symbolic | Thesis | Modules+bigram | Modules | Simple | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bacillus J≤0.6 (n=1621) | 0.035 | 0.125 | 0.159 | 0.281 | 0.269 | 0.281 | **0.450** | 0.314 |
| Pedobacter J≤0.6 (n=649) | 0.119 | 0.215 | 0.724 | 0.684 | 0.734 | 0.697 | **0.748** | 0.755 |
| Bacillus J≤0.5 (n=1465) | 0.034 | 0.136 | 0.176 | 0.318 | 0.292 | 0.322 | **0.473** | 0.351 |
| Pedobacter J≤0.5 (n=613) | 0.124 | 0.248 | 0.730 | 0.682 | 0.737 | 0.701 | **0.752** | 0.753 |

On both, **GenomeOcean remains last by a wide margin** and Simple is the strongest
or tied-strongest. Two folds is far too thin to conclude anything about method
ranking, but it does mean the genus split offers no counter-example to the
headline.

## 6. Caveats

- **Only 3 primary folds**, and 7 in total. Any median over them is extremely
  noisy; the per-fold numbers are the honest unit here.
- **The two dominant genera cannot be tested.** Xenorhabdus (75 % of constructs)
  and Photorhabdus (47 %) are excluded by the ≤45 % test cap, so this split says
  nothing about transfer to or from the best-represented organisms in the library.
- **Very low positive rates** on four folds (0.026–0.042), so their AP values sit
  close to the floor and their CIs are wide.
- **Erwinia should not be quoted** as evidence for sequence models without the
  Jaccard caveat above.
- Folds are not independent — all seven train on overlapping subsets of the same
  15,985 constructs.
