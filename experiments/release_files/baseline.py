#!/usr/bin/env python
"""Baseline for NRPS construct-production prediction.

Features: which modules a construct contains, plus which adjacent module pairs.
Model:    gradient-boosted trees.
No sequences and no embeddings -- the ordered module list is all it needs.

    python baseline.py constructs.csv

ASSUMED INPUT -- one CSV, three columns, one row per construct:

    modules    comma-separated module ids, IN ORDER
    produced   the label: True/False or 1/0
    split      "train" or "test"  (anything not "test" is trained on)

    modules,produced,split
    "1215,1216,105,1218,1219,1222",False,train
    "1215,1216,1232,1218,1219,1222",True,test

Module order matters: a bag of modules cannot tell apart two constructs built
from the same modules in a different order, and 264 constructs (1.65 %) are
exactly such permutation pairs carrying OPPOSITE labels. The adjacent-pair
features are what fix that.

The hyperparameters below were chosen once, on a different (random) split, and
then frozen -- there is no per-fold tuning here.

On our leave-one-family-out folds this lands within ~0.01 AP of the numbers we
published (e.g. TxlS 0.417 vs 0.409, XenS 0.403 vs 0.413, GxpS 0.482 vs 0.477),
comfortably inside the bootstrap CIs. The small gap is the vocabulary scope: this
script draws both vocabularies from every non-test row, whereas the published run
used every module id plus train-only bigrams. `run_all_folds.py` reproduces the
published table exactly if you need bit-identical numbers.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def features(module_lists, is_train):
    """Multi-hot over modules and over adjacent module pairs.

    Both vocabularies come from the TRAINING rows only, so a module or pair never
    seen in training gets no column at all -- the model abstains on it rather than
    extrapolating. That matters whenever the test set contains unfamiliar modules.
    """
    tokens = [set(ms) | set(zip(["^"] + ms, ms + ["$"])) for ms in module_lists]
    vocab = sorted({t for ts, tr in zip(tokens, is_train) if tr for t in ts}, key=str)
    col = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(tokens), len(vocab)), dtype=np.float32)
    for i, ts in enumerate(tokens):
        for t in ts:
            if t in col:
                X[i, col[t]] = 1.0
    return X


def main(path):
    d = pd.read_csv(path)
    modules = [[m.strip() for m in str(s).split(",")] for s in d["modules"]]
    y = d["produced"].astype(int).to_numpy()
    train = (d["split"] != "test").to_numpy()
    test = ~train

    X = features(modules, train)
    print(f"{train.sum()} train / {test.sum()} test rows, {X.shape[1]} features")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=400,
        early_stopping=True, validation_fraction=0.12, random_state=20260805)
    clf.fit(X[train], y[train])
    p = clf.predict_proba(X[test])[:, 1]

    # Average precision, against the only meaningful reference point: a model
    # that ranks at random scores the test set's own positive rate.
    chance = y[test].mean()
    print(f"average precision {average_precision_score(y[test], p):.3f}  "
          f"(chance {chance:.3f}, so lift "
          f"{average_precision_score(y[test], p) / chance:.2f}x)")
    print(f"ROC-AUC           {roc_auc_score(y[test], p):.3f}")
    return p


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "constructs.csv")
