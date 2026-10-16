#!/usr/bin/env python
"""Zero-shot / few-shot LLM evaluation on the two hardest folds.

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh

    # 1. write the prompts (no labels inside) and a separate answer key
    python experiments/family_llm.py emit  --folds TxlS XenS --n 400

    # 2a. score them yourself through whatever model access you have, OR
    # 2b. let the script call an API if a key is in the environment
    python experiments/family_llm.py run   --model claude-sonnet-4-5 --tag sonnet45

    # 3. turn responses into AP / ROC-AUC with bootstrap CIs
    python experiments/family_llm.py score --tag sonnet45

WHY THE DESIGN LOOKS LIKE THIS
------------------------------
**No agent, no tools.** Each construct is one stateless request. The model cannot
read `split_by_modules.csv` because it has no filesystem, not because it was
asked not to. That is the only leakage defence worth relying on.

**One construct per request.** Batching lets a model infer the base rate from its
own answers and calibrate to it, which would measure anchoring rather than
biology.

**The base rate is never stated.** Telling it "14 % produce" anchors the output.
AP and ROC-AUC only need a good *ranking*, so calibration is irrelevant here.

**The answer key never enters a prompt.** `emit` writes prompts.jsonl (no labels)
and key.csv (labels) as separate files.

**Few-shot examples come from TRAIN only.** Never from val or test -- that would
be the same leakage the whole family-split exercise exists to remove.

WHAT THE MODEL SEES
-------------------
Not module IDs -- those are internal identifiers, meaningless outside our
catalogue, and a model can only guess at them. Not raw sequence either: ~5,000
residues, which LLMs handle poorly. Instead each module is rendered from the
donor catalogue as type / architecture / specificity / organism / source BGC,
which is what a domain expert would actually look at:

    module 3: type=Elongation, architecture=TCDAT, specificity=Leu,
              organism=Xenorhabdus nematophila, BGC=XldS

The substantive question is whether the model knows NRPS assembly rules -- above
all **C-domain stereochemistry compatibility**, where `TCLAT` and `TCDAT` encode
LCL versus DCL condensation domains and a working fusion needs the donor
residue's stereochemistry to match. That is established NRPS engineering
knowledge, so a model that has it should beat chance with no training data at all.

HOW TO READ THE RESULT
----------------------
An LLM given descriptions has **strictly less information** than the `modules`
baseline, which sees module identity directly -- and our own result is that
essentially all the signal is in module identity. So losing to `modules+bigram`
is expected and uninformative. The comparisons that mean something:

    chance      the fold's positive rate -- does biological knowledge beat guessing?
    copytwin    the memorisation floor -- does it beat lookup with NO training data?
    thesis      a trained model on roughly the same information -- the fair match

Beating chance zero-shot would be genuinely notable. Beating `modules+bigram`
would be surprising enough that I would suspect contamination and check.

CONTROLS
--------
    --control shuffled   permute the labels at scoring time; AP must fall to chance
    --control scrambled  emit constructs assembled from RANDOM modules; a model
                         using real biology should score them lower than real ones
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifiers import bootstrap_ap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, "experiments", "splits")
OUT_DIR = os.path.join(REPO, "experiments", "results", "llm")
DONORS = os.path.join(REPO, "dataset", "donors_data.xlsx")
SEED = 20260805

SYSTEM = (
    "You are an expert in non-ribosomal peptide synthetase (NRPS) engineering. "
    "You judge whether an engineered chimeric NRPS assembly line will produce a "
    "detectable peptide product when expressed heterologously."
)

INSTRUCTION = """Below is an engineered NRPS construct, described module by module in order.

{body}

Will this construct produce a detectable peptide product?

Consider what is actually known to govern chimeric NRPS function, in particular
condensation-domain stereochemistry compatibility at each fusion point (LCL vs
DCL, visible in the architecture codes), domain-architecture compatibility
between neighbouring modules, and whether the modules come from compatible
biosynthetic origins.

Answer with ONE line and nothing else:
PROBABILITY: <integer 0-100>

where the integer is the percent chance it produces a product."""


def module_table() -> pd.DataFrame:
    don = pd.read_excel(DONORS)
    don["module number"] = don["module number"].astype(str).str.strip()
    return don.set_index("module number")


def describe(mods, mt) -> str:
    lines = []
    for i, m in enumerate(mods, 1):
        if m not in mt.index:
            lines.append(f"module {i}: (unknown module)")
            continue
        r = mt.loc[m]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        def g(k):
            v = r.get(k)
            return "unknown" if pd.isna(v) else str(v)
        lines.append(
            f"module {i}: type={g('module_type')}, architecture={g('module_architecture')}, "
            f"specificity={g('specificity')}, organism={g('organism')}, BGC={g('cluster')}")
    return "\n".join(lines)


def build_prompt(mods, mt, shots=()) -> str:
    parts = []
    if shots:
        parts.append("Here are labelled examples from the same engineering campaign:\n")
        for s_mods, s_lab in shots:
            parts.append(describe(s_mods, mt))
            parts.append(f"RESULT: {'produced a product' if s_lab else 'no product detected'}\n")
        parts.append("---\n")
    parts.append(INSTRUCTION.format(body=describe(mods, mt)))
    return "\n".join(parts)


# ----------------------------------------------------------------- emit
def cmd_emit(args) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    d = pd.read_csv(os.path.join(SPLIT_DIR, "split_by_modules.csv"))
    fs = pd.read_csv(os.path.join(SPLIT_DIR, "family_splits.csv"))
    mt = module_table()
    ml = [[t.strip() for t in str(s).split(",")] for s in d["modules"]]
    y = d["produced"].to_numpy().astype(int)
    rng = np.random.default_rng(SEED)

    prompts, key = [], []
    for fold in args.folds:
        sub = fs[fs["family"] == fold]
        if sub.empty:
            print(f"!!! fold {fold} not in family_splits.csv", file=sys.stderr)
            return 1
        sp = d["id"].map(dict(zip(sub["id"], sub["split"])))
        te = np.where((sp == "test").to_numpy())[0]
        tr = np.where((sp == "train").to_numpy())[0]

        # stratified sample that PRESERVES the fold's positive rate, so the AP is
        # comparable with the fold's own chance line
        pos, neg = te[y[te] == 1], te[y[te] == 0]
        n = min(args.n, len(te))
        n_pos = max(1, int(round(n * len(pos) / len(te))))
        n_pos, n_neg = min(n_pos, len(pos)), min(n - n_pos, len(neg))
        pick = np.concatenate([rng.choice(pos, n_pos, replace=False),
                               rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(pick)

        shots = ()
        if args.shots:
            # few-shot examples from TRAIN only, class-balanced
            tp, tn = tr[y[tr] == 1], tr[y[tr] == 0]
            k = args.shots // 2
            sel = list(rng.choice(tp, k, replace=False)) + list(rng.choice(tn, args.shots - k, replace=False))
            random.Random(SEED).shuffle(sel)
            shots = [(ml[i], int(y[i])) for i in sel]

        for i in pick:
            mods = ml[i]
            if args.control == "scrambled":
                allm = sorted({m for l in ml for m in l})
                mods = list(rng.choice(allm, len(mods), replace=False))
            prompts.append({"uid": f"{fold}:{int(d['id'][i])}", "fold": fold,
                            "system": SYSTEM, "prompt": build_prompt(mods, mt, shots)})
            key.append({"uid": f"{fold}:{int(d['id'][i])}", "fold": fold,
                        "produced": int(y[i]), "chance": float(y[te].mean()),
                        "n_test_full": len(te)})
        print(f"{fold}: sampled {len(pick)} of {len(te)} test constructs "
              f"({n_pos} producers, rate {n_pos/len(pick):.3f} vs fold {y[te].mean():.3f})")

    sfx = f"_{args.control}" if args.control else ""
    p = os.path.join(OUT_DIR, f"prompts{sfx}.jsonl")
    with open(p, "w") as fh:
        for r in prompts:
            fh.write(json.dumps(r) + "\n")
    k = os.path.join(OUT_DIR, f"key{sfx}.csv")
    pd.DataFrame(key).to_csv(k, index=False)
    print(f"\nwrote {p} ({len(prompts)} prompts, NO labels inside)")
    print(f"wrote {k}  <- the answer key, kept separate on purpose")
    print("\nExample prompt:\n" + "-" * 70)
    print(prompts[0]["prompt"][:1200])
    return 0


# ------------------------------------------------------------------ run
def call_openai(base, key, model, system, prompt, timeout=120):
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def call_anthropic(key, model, system, prompt, timeout=120):
    body = {"model": model, "max_tokens": 16, "temperature": 0, "system": system,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["content"][0]["text"]


def cmd_run(args) -> int:
    sfx = f"_{args.control}" if args.control else ""
    src = os.path.join(OUT_DIR, f"prompts{sfx}.jsonl")
    rows = [json.loads(l) for l in open(src)]
    dst = os.path.join(OUT_DIR, f"responses_{args.tag}{sfx}.jsonl")
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["uid"] for l in open(dst)}
        print(f"resuming: {len(done)} already scored")

    akey, okey = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not (akey or okey):
        print("No ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.\n"
              f"Score {src} through whatever access you have and write one JSON "
              f"object per line to {dst} with keys: uid, text", file=sys.stderr)
        return 2

    with open(dst, "a") as out:
        for i, r in enumerate(rows):
            if r["uid"] in done:
                continue
            try:
                txt = (call_anthropic(akey, args.model, r["system"], r["prompt"]) if akey
                       else call_openai(base, okey, args.model, r["system"], r["prompt"]))
            except Exception as e:
                print(f"  {r['uid']} FAILED {type(e).__name__}: {e}", flush=True)
                time.sleep(args.sleep * 4)
                continue
            out.write(json.dumps({"uid": r["uid"], "text": txt}) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
            time.sleep(args.sleep)
    print(f"wrote {dst}")
    return 0


# ---------------------------------------------------------------- score
def parse_prob(text):
    import re
    m = re.search(r"PROBABILITY\s*:\s*([0-9]{1,3})", str(text), re.I)
    if not m:
        m = re.search(r"\b([0-9]{1,3})\s*%", str(text))
    if not m:
        m = re.search(r"\b([0-9]{1,3})\b", str(text))
    if not m:
        return np.nan
    v = float(m.group(1))
    return v / 100.0 if 0 <= v <= 100 else np.nan


def cmd_score(args) -> int:
    sfx = f"_{args.control}" if args.control else ""
    key = pd.read_csv(os.path.join(OUT_DIR, f"key{sfx}.csv"))
    rp = args.responses or os.path.join(OUT_DIR, f"responses_{args.tag}{sfx}.jsonl")
    resp = [json.loads(l) for l in open(rp)]
    print(f"reading {rp}")
    r = pd.DataFrame(resp).drop_duplicates("uid", keep="last")
    r["score"] = r["text"].map(parse_prob)
    m = key.merge(r[["uid", "score"]], on="uid", how="left")
    bad = m["score"].isna().sum()
    if bad:
        print(f"!!! {bad}/{len(m)} responses unparseable -- filling with the sample mean")
        m["score"] = m["score"].fillna(m["score"].mean())

    rng = np.random.default_rng(SEED)
    rows = []
    for fold, g in m.groupby("fold"):
        y = g["produced"].to_numpy()
        s = g["score"].to_numpy()
        if args.control == "shuffled":
            y = rng.permutation(y)
        if y.sum() in (0, len(y)):
            print(f"  {fold}: degenerate labels, skipped")
            continue
        lo, hi = bootstrap_ap(y, s, n_boot=2000)
        rows.append(dict(family=fold, method=f"llm_{args.tag}" + (f"_{args.control}" if args.control else ""),
                         arm="frozen", test_ap=float(average_precision_score(y, s)),
                         ap_lo=lo, ap_hi=hi, test_roc_auc=float(roc_auc_score(y, s)),
                         chance_ap=float(y.mean()), n_test=len(y),
                         test_positives=int(y.sum()), n_unparsed=int(bad)))
        print(f"  {fold}: n={len(y)} pos={int(y.sum())} chance={y.mean():.3f} "
              f"AP={rows[-1]['test_ap']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"ROC-AUC={rows[-1]['test_roc_auc']:.3f}  lift={rows[-1]['test_ap']/y.mean():.2f}")
    out = pd.DataFrame(rows)
    dst = os.path.join(OUT_DIR, f"llm_results_{args.tag}{sfx}.csv")
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}")
    print("\ncompare against, on these folds (full test set, from results/family/all_methods_ci.csv):")
    ref = pd.read_csv(os.path.join(REPO, "experiments", "results", "family", "all_methods_ci.csv"))
    ref = ref[(ref.arm == "frozen") & ref.family.isin(out.family)]
    print(ref.pivot_table(index="family", columns="method", values="test_ap").round(3).to_string())
    return 0


# ---------------------------------------------------------------- export
TASK_MD = """# NRPS construct scoring task

This folder contains `prompts.jsonl`. Each line is one JSON object:

    {"uid": "...", "system": "...", "prompt": "..."}

**Your job:** answer every prompt and write `responses.jsonl` in this folder,
one JSON object per line:

    {"uid": "<the same uid>", "text": "PROBABILITY: 37"}

Rules:

1. Answer each prompt **on its own merits**. Do not try to make the answers add
   up to some expected number of positives -- there is no target rate, and
   guessing one will make the results worse.
2. Reply in the exact format the prompt asks for: `PROBABILITY: <0-100>`.
3. Use the full 0-100 range. Everything clustered at 20 or 50 carries no
   information and scores the same as random.
4. There is no answer key anywhere in this folder or anywhere you can reach.
   Do not go looking for one; just answer from what the prompt says.
5. Process every line. `responses.jsonl` must end up with the same number of
   lines as `prompts.jsonl`.

The prompts describe engineered non-ribosomal peptide synthetase (NRPS)
constructs module by module, and ask whether each will produce a detectable
peptide. Judge from the biology -- condensation-domain stereochemistry at the
fusion points, domain-architecture compatibility, and the compatibility of the
source biosynthetic gene clusters.
"""


def cmd_export(args) -> int:
    """Write a SELF-CONTAINED folder with prompts and no answer key.

    Intended for IDE / agent workflows (Cursor and similar) where you cannot set
    an API key. Point the export somewhere OUTSIDE this repository and open that
    folder as the workspace: an agent scoped to it then has no key.csv to find,
    which is a structural guarantee rather than an instruction it might ignore.
    """
    sfx = f"_{args.control}" if args.control else ""
    src = os.path.join(OUT_DIR, f"prompts{sfx}.jsonl")
    if not os.path.exists(src):
        print(f"!!! {src} missing -- run `emit` first", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in open(src)]
    dst = os.path.abspath(os.path.expanduser(args.dir))
    if dst.startswith(os.path.abspath(REPO)):
        print(f"!!! refusing to export inside the repository ({dst}).\n"
              f"    The whole point is that the workspace must not contain the "
              f"answer key. Pick a directory outside {REPO}.", file=sys.stderr)
        return 1
    os.makedirs(dst, exist_ok=True)

    if args.batch > 1:
        # one file per batch, for pasting into a chat window
        bdir = os.path.join(dst, "batches")
        os.makedirs(bdir, exist_ok=True)
        for b in range(0, len(rows), args.batch):
            chunk = rows[b:b + args.batch]
            txt = [f"Score each of the {len(chunk)} NRPS constructs below independently.",
                   "Reply with exactly one line per construct, nothing else:",
                   "<uid> PROBABILITY: <0-100>", ""]
            for r in chunk:
                txt.append(f"### {r['uid']}")
                txt.append(r["prompt"])
                txt.append("")
            open(os.path.join(bdir, f"batch_{b//args.batch:03d}.md"), "w").write("\n".join(txt))
        print(f"wrote {len(rows)//args.batch + 1} batch files to {bdir}")

    with open(os.path.join(dst, "prompts.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    open(os.path.join(dst, "TASK.md"), "w").write(TASK_MD)
    open(os.path.join(dst, "responses.jsonl"), "a").close()

    leaked = [f for f in os.listdir(dst) if "key" in f.lower()]
    print(f"\nexported {len(rows)} prompts to {dst}")
    print(f"  TASK.md          instructions for the model")
    print(f"  prompts.jsonl    the prompts, no labels")
    print(f"  responses.jsonl  empty, to be filled in")
    print(f"  answer-key files present in the export: {leaked if leaked else 'NONE (correct)'}")
    print(f"\nThe key stays at {os.path.join(OUT_DIR, f'key{sfx}.csv')} -- outside that folder.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.set_defaults(fn=cmd_emit)
    e.add_argument("--folds", nargs="+", default=["TxlS", "XenS"],
                   help="default: the two hardest folds by median max-Jaccard")
    e.add_argument("--n", type=int, default=400, help="constructs sampled per fold")
    e.add_argument("--shots", type=int, default=0, help="few-shot examples, from TRAIN only")
    e.add_argument("--control", choices=("scrambled",), default=None)
    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--sleep", type=float, default=0.2)
    r.add_argument("--control", choices=("scrambled",), default=None)
    x = sub.add_parser("export"); x.set_defaults(fn=cmd_export)
    x.add_argument("--dir", required=True, help="a directory OUTSIDE this repo")
    x.add_argument("--batch", type=int, default=0, help="also write paste-sized batch files")
    x.add_argument("--control", choices=("scrambled",), default=None)
    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--tag", required=True)
    s.add_argument("--responses", default=None, help="path to responses.jsonl (default: results/llm/)")
    s.add_argument("--control", choices=("scrambled", "shuffled"), default=None)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
