Hey Usman,

The baseline just feeds which modules the construct contains, plus which adjacent
module pairs, into a gradient-boosted tree. So the features come from the module list
alone.

Two kinds of binary feature:

- **`modules`** — one column per module id, 1 if the construct uses it. Order-blind.
- **`bigrams`** — one column per adjacent module pair, with `^` / `$` sentinels so
  "which module is first / last" is captured too.

Both vocabularies are built on the training rows only, so a module or pair unseen in
training gets no column at all. We have done this on purpose, to keep the baseline as simple as possible and get a true "lower bound".

I have attached a small code snippet, which I hope shows what it does well enough for reporduction.

The hyperparameters in there were picked once on an earlier random split and then frozen. The
tree settings themselves do not really matter much, i.e. the four learning-rate / leaf-count
combinations we swept land within 0.017 AP of each other on validation. What does seem to
matter is the model family (logistic regression and MLPs on the same features spread
over 0.17 AP), so I would not read the tree choice as arbitrary.

```python
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

d = pd.read_csv("constructs.csv")      # columns: modules, produced, split
modules = [[m.strip() for m in str(s).split(",")] for s in d["modules"]]
y = d["produced"].astype(int).to_numpy()
train = (d["split"] != "test").to_numpy()

# one token per module, plus one per adjacent pair; '^'/'$' mark first/last
tokens = [set(ms) | set(zip(["^"] + ms, ms + ["$"])) for ms in modules]
vocab = sorted({t for ts, tr in zip(tokens, train) if tr for t in ts}, key=str)
col = {t: j for j, t in enumerate(vocab)}

X = np.zeros((len(tokens), len(vocab)), dtype=np.float32)
for i, ts in enumerate(tokens):
    for t in ts:
        if t in col:
            X[i, col[t]] = 1.0

clf = HistGradientBoostingClassifier(
    learning_rate=0.1, max_leaf_nodes=63, max_iter=400,
    early_stopping=True, validation_fraction=0.12, random_state=20260805)
clf.fit(X[train], y[train])
p = clf.predict_proba(X[~train])[:, 1]
```

Best,
Tobias
