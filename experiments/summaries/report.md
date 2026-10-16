# BGC production prediction

The main goal was to find a hard train test split such that pure module knowledge would not work. However, whatever we have tried the simple module + bigram baseline always performs best. Leaving the question if that is the classifier we maybe should go with. 

---

## 1. Train / val / test split

Our earlier result (test AP 0.734) came from a **random** train/test split. This was too optimistic, because every construct is assembled from the
same pool of 1,173 modules, 93 % of test constructs had a training construct
differing by exactly one module in one position. The model was mostly
recognising something it had almost already seen.

Too measure how much of our success came from this, we used as a predictor the label of the nearest
one-module-swap neighbour. This scores AP 0.534 against our
reported 0.734. So roughly three-quarters of the classifier could have been simple lookup. (Worth
noting: 18.3 % of one-module-swap pairs carry opposite labels,
so a single module swap flips the outcome about one time in five. I am not sure if this is a measurement error or that one module swap can change that much.)

We therefore split by biosynthetic family instead. Every module is tagged
with the wild-type BGC it came from (GxpS, SzeS, XbcS…). We pick one family,
delete all its modules from training, and test on every construct using at
least one of them. 29 such folds are usable (filtered by size and number of producing sequences). We also built two variants: holding out three
families at once (26 folds) and holding out a whole bacterial genus
(7 folds). All splits are defined purely from the module catalogue, never from
the labels.

To find the most difficult splits, we initially ranked folds by that
"one-module-swap" statistic, but found this to be unreliable: it only compares constructs of
equal length in matched positions, so it is blind to insertions and deletions. Hence, we used the jaccard distance for ranking. Specifically, for every construct in a fold's test set
we find its most similar training construct, scored as Jaccard overlap of the
two module sets ((modules in common) / (modules in either)) and take the
median of that across the fold (med J). A low value means the
typical test construct has no close relative in training, i.e. a hard fold.

For two six-module constructs the scale reads:

| modules shared (of 6) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| J | 0.09 | 0.20 | 0.33 | 0.50 | 0.71 | 1.00 |

So on TxlS (med J 0.20, our hardest fold) the typical test construct shares
only 2 of its 7 modules with anything in training. On XfaS, which the
old one-swap statistic scored as perfectly clean, it shares 5 of 6. That
gap is the whole difference between the two measures, and why we switched.
Across all 62 folds med J runs from 0.20 to 0.75.

Being set-based, Jaccard treats substitutions, insertions, deletions and
reorderings alike; the one-swap statistic only saw substitutions at matched
length. Ranking by med J and reporting on the hardest folds is now how we judge
a result.


## 2. Models compared

| Model | What it sees |
|---|---|
| **Modules** | 1,173 OHC vector which modules are present. No sequence, no order. |
| **Modules+bigram** | The same, plus which module follows which (order). |
| **GenomeOcean** | A frozen 4-billion-parameter DNA foundation model, finetuned on 11 M BGCs; we train a small classifier on its aggregated embeddings. |
| **Symbolic** | LLM-designed protein features (composition, known NRPS motifs, 6-mer counts). No pretrained model. |
| **Thesis** | Usman's Thesis: source organism, domain-architecture compatibility, GC content, and k-mer counts → gradient-boosted trees. |
| **Simple** | Our own: Modules+bigram plus four numbers from Thesis (architecture compatibility, GC median/max). Built to test which parts of Thesis matter. |
| **Ensemble** | Thesis blended with Modules+bigram (mixing weight chosen on validation). |
| Copy-twin | The "copy your nearest neighbour" memorisation floor |

## 3. Results on the hardest folds

All 62 folds from the three split schemes, ranked by Jaccard similarity to
training (lower = harder). Test Average Precision (AP) and chance is that fold's own producer rate.

| # | Fold | med J | n | chance | *Copy-twin* | GenomeOcean | Symbolic | Thesis | Mod+bigram | Modules | Simple | Ensemble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TxlS | 0.20 | 2389 | 0.136 | *0.230* | 0.361 | 0.331 | 0.363 | **0.409** | 0.402 | 0.401 | 0.397 |
| 2 | XenS | 0.22 | 871 | 0.090 | *0.361* | 0.073 | 0.077 | 0.380 | 0.413 | **0.424** | 0.396 | 0.398 |
| 3 | XbcS | 0.33 | 2347 | 0.084 | *0.290* | 0.188 | 0.142 | 0.358 | 0.269 | 0.286 | **0.393** | 0.363 |
| 4 | FldS+TxlS+tri | 0.33 | 2909 | 0.130 | *0.245* | 0.362 | 0.405 | 0.430 | **0.480** | 0.452 | 0.435 | 0.467 |
| 5 | FldS | 0.33 | 505 | 0.103 | *0.671* | 0.632 | 0.746 | 0.790 | 0.792 | 0.762 | 0.799 | **0.811** |
| 6 | XtvB | 0.38 | 464 | 0.140 | *0.152* | 0.387 | 0.417 | 0.473 | 0.439 | 0.399 | 0.328 | **0.531** |
| 7 | GxpS+apici+tolaa | 0.40 | 6214 | 0.146 | *0.184* | 0.284 | 0.378 | 0.494 | 0.485 | 0.442 | 0.455 | **0.520** |
| 8 | Bacillus | 0.40 | 1948 | 0.042 | *0.149* | 0.113 | 0.155 | 0.224 | 0.271 | 0.297 | **0.337** | 0.291 |
| | **median, 10 hardest** | | | *0.117* | *0.207* | 0.285 | 0.352 | 0.405 | **0.426** | 0.413 | 0.398 | 0.433 |

The same folds by **ROC-AUC** (0.5 = coin flip, so unlike AP it is directly
comparable across folds):

| # | Fold | *Copy-twin* | GenomeOcean | Symbolic | Thesis | Mod+bigram | Modules | Simple | Ensemble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TxlS | *0.640* | 0.782 | 0.767 | 0.788 | 0.811 | 0.807 | **0.828** | 0.811 |
| 2 | XenS | *0.792* | **0.360** | 0.413 | 0.851 | 0.850 | 0.865 | 0.852 | **0.868** |
| 3 | XbcS | *0.756* | 0.750 | 0.606 | 0.862 | 0.810 | 0.811 | 0.853 | **0.866** |
| 4 | FldS+TxlS+tri | *0.666* | 0.781 | 0.795 | 0.828 | **0.852** | 0.843 | 0.847 | 0.851 |
| 5 | FldS | *0.822* | 0.956 | 0.935 | 0.970 | 0.962 | 0.962 | 0.964 | **0.971** |
| 6 | XtvB | *0.499* | 0.820 | **0.845** | 0.804 | 0.810 | 0.804 | 0.732 | 0.822 |
| 7 | GxpS+apici+tolaa | *0.560* | 0.760 | 0.833 | 0.841 | 0.820 | 0.800 | 0.824 | **0.853** |
| 8 | Bacillus | *0.661* | 0.666 | 0.796 | 0.877 | 0.863 | 0.861 | **0.894** | 0.890 |
| | **median, 10 hardest** | *0.664* | 0.771 | 0.814 | 0.840 | 0.835 | 0.827 | 0.849 | **0.852** |

ROC-AUC tells a slightly softer version of the same story: the top five methods
sit at 0.83–0.85 and GenomeOcean trails at 0.771, but the gap is smaller than in
AP because ROC-AUC weights all thresholds equally while AP weights the top of the
ranking — and the top of the ranking is what matters if you are choosing which
constructs to build. The XenS row is the clearest single result in the
report: GenomeOcean scores 0.360, i.e. it ranks producers worse than random,
while the module-based methods are at 0.85–0.87 on the same constructs.

**Using Jaccard to define folds does not work.** We tried both
alternatives and neither is better. Grouping constructs by Jaccard fails on our data. At a threshold of 0.7 it still leaves 44 % of the near-duplicates in place (it only catches one-module swaps for constructs with
≥6 modules, and 56 % of ours are smaller), while lowering it to 0.6 chains
everything into one group holding 53 % of the data. Capping the test set by
Jaccard does work mechanically, but the cap turns out to be correlated with the
outcome (mean |r| = 0.16, sign flipping between folds), so it reshapes which
constructs get tested rather than just removing contamination. We ran it anyway
at three thresholds and the ranking barely moves (rank correlation 0.8–1.0 against
the full test set) and GenomeOcean is last at every one while module + bigram stays best, so it seems that the conclusions do not depend on this choice, and we keep the simpler option of ranking folds and
reporting on the full test set.

## 4. Takeaway

Modules+bigram, Modules, Thesis, Simple and the Ensemble all sit within a narrow band
(AP 0.40–0.43 on the hardest folds) and are statistically indistinguishable from
each other in paired tests (we ommitted the Confidence intervall in the table for readability). Nothing we tried beats a model that knows only which
modules a construct contains and in what order.

**The DNA foundation model does not earn its place here.** GenomeOcean is the
weakest method on the family split, and at every similarity
threshold. It scores 0.285 AP on the hardest folds, only 0.078 above the memorisation
floor. On the second-hardest fold it ranks producers even worse than random.

**Caveats.** Folds are ofc. not independent. The overlapp between the folds range between 27%-87%. 
