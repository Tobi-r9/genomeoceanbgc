# Is the 18.3 % flip rate biology rather than noise?

**A colleague's hypothesis, the test they proposed, and what the data says.**

Date: 2026-09-08 · Data: `experiments/splits/split_by_modules.csv` (15,985 constructs)
Reproduce: the three analyses in this note are self-contained pandas/numpy over that
one CSV; every number below is quoted with the sample size it came from.

---

## 0. The number we are arguing about

First, a reminder of where **18.3 %** comes from, because everything below depends on it.

Take every pair of constructs in our dataset that differ in **exactly one position** —
same length, same modules everywhere else, one single module swapped. Call these
**one-swap pairs**. There are **157,170** of them.

Now ask: how often do the two members of such a pair have *opposite* labels — one
produces the compound, the other does not?

> **18.27 %** of one-swap pairs are label-flipped. (95 % CI 18.10 – 18.46 %)

We originally raised this as a **warning about evaluation**. If you put one member of
such a pair in the training set and the other in the test set, a model can score well by
memorising "constructs that look like this one produce" without understanding anything.
Roughly 93 % of the test set in a random split has a near-twin in training, which is why
the random-split score (AP 0.734) collapses to 0.40 – 0.43 on our leave-one-family-out
split. That argument is about **leakage**, and it does not depend on *why* the 18.3 % of
flips happen.

The colleague's reply is about the **why**. That is a different and genuinely interesting
question, so it is worth answering properly.

---

## 1. Their concern

Their argument, restated as fairly as I can:

> *You are treating the 18.3 % as if it were noise — as if two nearly identical
> constructs "should" have the same label and the fact that 18 % of them don't is a
> problem with the data. But that is exactly what NRPS biology predicts. An NRPS
> assembly line only works if **every junction between adjacent modules works**. Swap one
> module and you break **two** junctions (the one before it and the one after it). Each
> junction has some independent chance of being compatible. So a single swap genuinely
> flips the outcome a large fraction of the time. 18 % is not noise — it is the
> signature of multiplicative junction compatibility.*

Formally, the model they are proposing is:

$$P(\text{construct produces}) = p^{\,(k-1)}$$

where $k$ is the number of modules, $k-1$ is the number of junctions between them, and
$p$ is the probability that any one junction is compatible, assumed **independent** across
junctions.

**This is a serious hypothesis, not a hand-wave.** It is exactly how people think about
combinatorial NRPS libraries, and it makes the 18.3 % sound like a feature of the system
rather than a defect of the dataset.

### Their arithmetic is correct

I checked it. Calibrate $p$ so the model reproduces our overall production rate of
14.1 % at the typical construct length of 6 modules (5 junctions):

$$p = 0.141^{1/5} = 0.676$$

Now work out how often a one-swap pair should be label-flipped. Write the pair as A and B.
They share all the junctions that the swap did not touch; call the probability that all of
those work $S$. A produces if $S$ **and** A's own new junctions work; likewise for B.

$$P(\text{flip}) = 2 \cdot S \cdot p^{m}\,(1 - p^{m})$$

where $m$ is the number of junctions the swap changed: **$m = 2$ for an internal swap**,
**$m = 1$ for a terminal swap** (the first or last module has only one neighbour).

For a 6-module construct:

| scenario | formula | predicted flip rate |
|---|---|---|
| internal swap (2 junctions changed) | $2 \cdot 0.141 \cdot (1 - 0.676^2)$ | **15.3 %** |
| terminal swap (1 junction changed) | $2 \cdot 0.141 \cdot (1 - 0.676)$ | **9.1 %** |
| labels carry no information at all | $2 \cdot 0.141 \cdot 0.859$ | **24.2 %** |

Their claimed range — *"12–20 % for internal swaps, less for terminal, ~24 % if the
sequence were uninformative"* — is **exactly right**.

And note where our observed 18.3 % falls: **between 15.3 % and 24.2 %**. So at the level
of the single headline number, **their model is consistent with the data.** You cannot
refute it by pointing at 18.3 % alone. That is precisely why a sharper test is needed.

---

## 2. Why they propose a test (and what it is supposed to show)

The problem with "18.3 % is consistent with my model" is that 18.3 % is *also* consistent
with a lot of other explanations — measurement noise, unmodelled chemistry, sub-library
effects, plain unpredictability. One number cannot distinguish between them.

So the colleague proposes looking for a **gradient**: a pattern that the multiplicative
junction model predicts and that competing explanations do not. If the flip rate is really
produced by breaking independent junctions, then the flip rate must **depend on how many
junctions you broke**.

That gives two concrete, falsifiable predictions.

### Prediction A — internal swaps must flip more often than terminal swaps

An internal swap breaks 2 junctions; a terminal swap breaks 1. From the formula above, the
ratio is

$$\frac{P(\text{flip} \mid \text{internal})}{P(\text{flip} \mid \text{terminal})}
= \frac{1 - p^{2}}{1 - p} = 1 + p = \mathbf{1.68}$$

**This is the good test, and it is worth being explicit about why.** The ratio

- does **not** depend on the construct length $k$ — the shared-junction term $S$ cancels;
- does **not** depend on how well calibrated $p$ is — any $p$ between 0.5 and 0.9 gives a
  ratio between 1.5 and 1.9;
- is a **within-length, within-library** comparison, so it is not confounded by which
  sub-library a construct came from.

In other words: it is a clean, assumption-light prediction. If junction breakage drives
the flips, internal swaps must flip roughly **1.7× more often** than terminal ones.
If flips come from something that does not care about junction counts, the two will be
about the **same**.

### Prediction B — production rate must decay geometrically with length

The same model says a construct with $k$ modules produces with probability $p^{k-1}$.
Every extra module is another independent chance to fail. With $p = 0.676$ that is a steep
decay: about **31 % at 4 modules, 21 % at 5, 14 % at 6, 9.5 % at 7, 6.4 % at 8**.

**What the tests are supposed to show.** If both gradients are there, the colleague is
right: the 18.3 % is structured biology, we should stop calling it noise, and — more
usefully — we should be modelling **junctions**, not modules. If the gradients are absent,
then whatever is generating the flips is not "independent junctions multiplying together",
and the 18.3 % stays what we treated it as: variation the module identities do not
explain.

---

## 3. The answer

I ran both tests on all **157,170** one-swap pairs. **Neither gradient is there.**

### Test A — internal vs. terminal: no difference

Headline, pooling everything:

| swap type | junctions broken | pairs | flip rate | 95 % CI |
|---|---|---|---|---|
| internal | 2 | 140,272 | **18.26 %** | 18.07 – 18.46 % |
| terminal | 1 | 16,898 | **18.30 %** | 17.72 – 18.86 % |

Predicted ratio **1.68**. Observed ratio **1.00**.

The pooled comparison mixes construct lengths, so here it is broken out per length, which
is the comparison that actually controls for everything:

| modules $k$ | internal flip (pairs) | terminal flip (pairs) | observed ratio | predicted |
|---|---|---|---|---|
| 4 | 15.17 % (29,635) | 13.04 % (8,835) | **1.16** | 1.68 |
| 5 | 27.92 % (29,998) | 25.19 % (7,543) | **1.11** | 1.68 |
| 6 | 11.12 % (51,760) | 12.03 % (291) | **0.92** | 1.68 |
| 7 | 24.47 % (27,658) | 1.52 % (132) | 16.15 | 1.68 |
| 8 | 22.55 % (927) | 3.12 % (64) | 7.21 | 1.68 |

The three well-powered rows — $k = 4, 5, 6$, which hold **98.6 %** of all terminal pairs —
give ratios of **1.16, 1.11 and 0.92**. The prediction is 1.68. There is a hint of an
effect at $k=4$ and $k=5$, but it is roughly a quarter of the predicted size, and it
reverses at $k=6$.

**The two rows that appear to support the hypothesis are artefacts of tiny samples.**
The 16× at $k=7$ comes from **2 flipped pairs out of 132**. The 7× at $k=8$ comes from
**2 flipped pairs out of 64**. Two events cannot carry this argument.

One thing worth checking, and I did: is the terminal comparison even fair? It is. The
first position is a **Starter** module in 15,676 of 15,985 constructs and the last is a
**Release** or **Dimodule** in essentially all of them. So terminal swaps substitute
like-for-like — Starter for Starter, Release for Release — and are not secretly a
different kind of perturbation.

### Test B — no geometric decay with length

| modules $k$ | constructs | observed production rate | multiplicative model predicts |
|---|---|---|---|
| 3 | 72 | 0.097 | 0.457 |
| 4 | 3,904 | 0.105 | 0.309 |
| 5 | 4,757 | **0.199** | 0.209 |
| 6 | 4,120 | 0.092 | 0.141 |
| 7 | 2,561 | **0.174** | 0.095 |
| 8 | 402 | 0.109 | 0.064 |

The predicted column falls smoothly by a factor of ~1.5 at every step. The observed column
**zigzags and does not decay at all**: 7-module constructs produce at 17.4 %, *more often*
than 4-module constructs at 10.5 %. Three-module constructs — which the model says should
be the easiest thing in the library at 45.7 % — produce at 9.7 %, the second *lowest* rate
in the table.

The flip rate shows the same non-pattern: **6.1 %, 14.7 %, 27.4 %, 11.1 %, 24.4 %, 21.3 %**
for $k = 3 \ldots 8$. Up, up, down, up, down.

### An honest caveat on Test B

Construct length is partly tied to which BGC family a construct comes from, so Test B is
not a perfectly controlled comparison. I checked how bad this is: at $k = 4, 5, 6$ the
single most common family accounts for only **16 %, 14 % and 28 %** of constructs of that
length, so those lengths are well mixed across families and the comparison is meaningful
there. At $k = 7, 8$ one family accounts for 47 %, so those two rows are the confounded
ones — and they are the rows that deviate *upward*, i.e. against the model.

This is why **Test A is the decisive one**. It compares swaps *within the same construct
length and the same library*, so no such confound exists, and it is where the prediction
fails most cleanly.

### Summary of the verdict

| claim | status |
|---|---|
| Their arithmetic (15.3 % / 9.1 % / 24.2 %) | ✅ correct |
| 18.3 % is *consistent* with the multiplicative model | ✅ true — cannot be refuted by that number alone |
| Internal swaps flip ~1.7× more than terminal | ❌ **fails** — observed 1.16 / 1.11 / 0.92 on 98.6 % of the data |
| Production rate decays as $p^{k-1}$ | ❌ **fails** — observed rate does not decay, it zigzags |

**The specific mechanism they proposed — independent junctions multiplying together — is
ruled out by their own test.** They designed the right experiment; it just came back
negative.

---

## 4. What this does and does not mean

Three things worth being careful about, so this doesn't get over-read in either direction.

**It does not mean the flips are "just noise".** Ruling out one mechanistic model is not
the same as showing there is no biology. Junction compatibility could easily be real but
**not independent and not uniform** — some junctions matter enormously, most don't matter
at all, and the effect is dominated by a handful of specific domain interfaces. That model
would produce no length gradient and no internal/terminal gradient, and it is entirely
compatible with everything above. What the data rules out is the *simple multiplicative*
version, which is the version that predicts the gradients.

**It does not weaken the leakage argument, and it never could have.** This is the point I
would emphasise in a reply. Our concern was about **evaluation**: if a test construct has a
one-module twin in the training set, a model gets credit for copying rather than
generalising. That is true whether the flips are biology or noise — arguably *more*
troubling if they are biology, because then a model that memorises twins is failing to
learn a real and learnable signal. The 18.3 % is why nearest-neighbour copying only gets
AP 0.534 on a random split instead of ~1.0; the *reason* for the 18.3 % does not enter that
argument. So there is no disagreement to resolve here — the two points are about different
things.

**The negative result is itself informative.** "Production is not a product of independent
per-junction probabilities" is a real finding about this library, and it explains something
we had observed but not accounted for: why construct length is such a weak feature in every
model we have fitted. It should not be, if the multiplicative picture held.

---

## 5. The part of their suggestion that we should actually act on

Underneath the mechanistic claim there is a **modelling suggestion**, and that one is good
and largely independent of whether the multiplicative model holds:

> *Describe the junctions, not just the modules.*

Two honest notes on this.

**Partly already done.** Our `modules+bigram` feature block encodes exactly which
**adjacent module pairs** occur in a construct — that is a junction feature. It is our
joint-best method (AP 0.40 – 0.43 on the family split), and it does beat plain `modules`,
so junctions do carry information. This part of their intuition is supported.

**The genuinely new part.** A module bigram is an *identity* — "module 47 followed by
module 12". It is useless for a junction that never appeared in training, which is exactly
the situation on our leave-one-family-out folds, and it is the main reason those folds are
hard. What they are really pointing at is a **junction descriptor**: not *which* modules
meet, but *what kind of interface* it is — which amino-acid residue is being handed over,
which condensation-domain subtype receives it, what the donor/acceptor specificity is. Two
junctions that never co-occurred in training but share a descriptor would then be treated
as similar.

That is a real, testable idea and it targets the exact failure mode we measured. It is the
most promising modelling direction to come out of this exchange, and it is worth building
even though the mechanistic hypothesis behind it did not survive.

---

## 6. Suggested reply, in three sentences

> Your arithmetic is right — I checked it, and 18.3 % does sit inside the range the
> multiplicative model predicts, so that number alone can't distinguish your explanation
> from noise. I ran the gradient test you proposed on all 157k one-swap pairs and it comes
> back negative: internal swaps flip 18.26 % and terminal swaps 18.30 %, versus the 1.68×
> ratio the model predicts, and per-length the ratios are 1.16 / 1.11 / 0.92 on the 98.6 %
> of terminal pairs where we have power; production rate also doesn't decay with length
> (17.4 % at 7 modules vs 10.5 % at 4). Separately, the leakage point stands either way —
> it's about a test construct having a one-swap twin in training, which is a problem
> whether the flip is biology or noise — but your underlying suggestion to model
> **junction descriptors** rather than module identities is the best idea we have for the
> unseen-module folds, and I'd like to build it.

---

## Related documents

- [`../FAMILY_SPLIT.md`](../FAMILY_SPLIT.md) — how the leave-one-family-out split is built
- [`../FAMILY_RESULTS.md`](../FAMILY_RESULTS.md) — all methods on all 29 folds
- [`../FAMILY_GUIDE.md`](../FAMILY_GUIDE.md) — plain-English walkthrough, including the
  one-swap twin-rate analysis
- [`01_jaccard_capped_split.md`](01_jaccard_capped_split.md) — fold-strictness ranking
- [`report.md`](report.md) — the two-page supervisor summary
