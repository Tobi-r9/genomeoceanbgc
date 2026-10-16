#!/usr/bin/env python
"""Extract frozen GenomeOcean-4B-bgcFM embeddings for the BGC constructs.

One process per GPU; shards are disjoint and each writes its own file:

    python experiments/embed_dataset.py --shard 0 --num-shards 4

Outputs $GO_SCRATCH/bgc_clf/emb/shard{i}of{n}.npz with `ids`, `mean`, `last`,
`n_tokens`, `truncated`.

Pooling: `mean` is the mask-weighted mean over token positions -- the same
convention as `genomeocean.llm_utils.LLMUtils.predict`, so these embeddings are
comparable to the package's. `last` is the final real token's hidden state,
which for a causal decoder has attended to the whole sequence; it costs nothing
extra to save and sometimes beats mean-pooling.

The model is loaded with trust_remote_code=False. This is REQUIRED on
transformers 5.x -- the checkpoint's bundled 4.38-era modeling code silently
computes wrong hidden states. See CLAUDE.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, PreTrainedTokenizerFast

MAX_TOKENS = 10240          # GenomeOcean's usable context
HIDDEN = 3072

SCRATCH = os.path.join(os.environ.get("GO_SCRATCH", "/e/scratch/profound/thoeppe/genomeocean"),
                       "bgc_clf")


def build_batches(lengths, max_batch_tokens, max_batch_size):
    """Group length-sorted indices into batches bounded by padded token count."""
    order = np.argsort(lengths, kind="stable")
    batches, cur = [], []
    for i in order:
        trial = cur + [i]
        # cost of the padded batch is len(batch) * longest member
        if cur and (len(trial) * lengths[trial].max() > max_batch_tokens
                    or len(trial) > max_batch_size):
            batches.append(np.array(cur))
            cur = [i]
        else:
            cur = trial
    if cur:
        batches.append(np.array(cur))
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-batch-tokens", type=int, default=65536)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N sequences")
    ap.add_argument("--out-dir", default=os.path.join(SCRATCH, "emb"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard{args.shard}of{args.num_shards}.npz")
    if os.path.exists(out_path):
        print(f"{out_path} exists, nothing to do")
        return 0

    model_dir = os.environ["GO_BGC_MODEL"]
    df = pd.read_parquet(os.path.join(SCRATCH, "dataset_split.parquet"),
                         columns=["id", "sequence"])
    if args.limit:
        df = df.iloc[:args.limit]
    # Contiguous, disjoint, order-independent sharding.
    df = df.iloc[args.shard::args.num_shards].reset_index(drop=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(df)} sequences", flush=True)

    tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
    tok.model_max_length = MAX_TOKENS
    tok.padding_side = "right"      # real tokens keep positions 0..n-1

    print("tokenising...", flush=True)
    enc = tok(df["sequence"].tolist(), truncation=True, max_length=MAX_TOKENS,
              padding=False)["input_ids"]
    lengths = np.array([len(e) for e in enc])
    truncated = lengths >= MAX_TOKENS
    print(f"tokens: median {int(np.median(lengths))}, max {lengths.max()}, "
          f"truncated {int(truncated.sum())}", flush=True)

    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to("cuda").eval()
    model.config.use_cache = False

    mean_out = np.zeros((len(df), HIDDEN), dtype=np.float32)
    last_out = np.zeros((len(df), HIDDEN), dtype=np.float32)

    batches = build_batches(lengths, args.max_batch_tokens, args.max_batch_size)
    print(f"{len(batches)} batches", flush=True)
    pad_id = tok.pad_token_id or 0
    t0 = time.time()
    done = 0
    for bi, batch in enumerate(batches):
        L = int(lengths[batch].max())
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, j in enumerate(batch):
            n = lengths[j]
            ids[r, :n] = torch.tensor(enc[j], dtype=torch.long)
            mask[r, :n] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        idx = (mask.sum(1) - 1).clamp(min=0)
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        mean_out[batch] = pooled.float().cpu().numpy()
        last_out[batch] = last.float().cpu().numpy()
        done += len(batch)
        if bi % 25 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)}  {done}/{len(df)} seqs  "
                  f"{el:.0f}s  {done/max(el,1e-9):.1f} seq/s  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)

    assert np.isfinite(mean_out).all() and np.isfinite(last_out).all(), "non-finite embedding"
    np.savez(out_path, ids=df["id"].to_numpy(), mean=mean_out, last=last_out,
             n_tokens=lengths, truncated=truncated)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
