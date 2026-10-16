#!/usr/bin/env python
"""Smoke test for GenomeOcean-4B-bgcFM on JUPITER (aarch64 / GH200).

    source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
    python jupiter/test_bgc_model.py                 # tokenizer + forward + embed + ppl
    python jupiter/test_bgc_model.py --with-vllm     # also vLLM generation (needs ~25 GB HBM)

Every check is independent and reports PASS/FAIL rather than aborting, so one
broken path does not hide the state of the others. Exit code is non-zero if any
check failed.
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

MODEL = os.environ.get(
    "GO_BGC_MODEL",
    "/e/scratch/profound/thoeppe/genomeocean/hf_models/GenomeOcean-4B-bgcFM",
)
# GenomeOcean's tokenizer is byte-pair over nucleotides, ~4 bp per token.
MAX_LEN = 10240
HIDDEN = 3072          # config.json hidden_size
VOCAB = 4096           # config.json vocab_size

SEQS = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA",
]

RESULTS = []


def check(name):
    """Decorator: run a check, record PASS/FAIL, never propagate."""
    def deco(fn):
        print(f"\n{'=' * 72}\n>>> {name}\n{'=' * 72}")
        t0 = time.time()
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
            print(f"--- PASS ({time.time() - t0:.1f} s) {detail or ''}")
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
            print(f"--- FAIL ({time.time() - t0:.1f} s) {type(e).__name__}: {e}")
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-vllm", action="store_true",
                    help="also test vLLM generation (allocates ~25 GB of HBM)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.25,
                    help="vLLM gpu_memory_utilization (default 0.25 = ~24 GB of a GH200)")
    args = ap.parse_args()

    print(f"model            {MODEL}")
    print(f"torch            {torch.__version__} | cuda {torch.version.cuda}")
    import transformers
    print(f"transformers     {transformers.__version__}")
    print(f"cuda available   {torch.cuda.is_available()} | devices {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device 0         {torch.cuda.get_device_name(0)}")

    from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                              PreTrainedTokenizerFast)

    state = {}

    # ---------------------------------------------------------------- config
    @check("config loads (native mistral, no remote code)")
    def _cfg():
        cfg = AutoConfig.from_pretrained(MODEL)
        assert cfg.model_type == "mistral", cfg.model_type
        assert cfg.hidden_size == HIDDEN, cfg.hidden_size
        assert cfg.vocab_size == VOCAB, cfg.vocab_size
        state["cfg"] = cfg
        return (f"{cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
                f"heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} kv, vocab {cfg.vocab_size}")

    # ------------------------------------------------------------- tokenizer
    @check("tokenizer (PreTrainedTokenizerFast) encodes DNA")
    def _tok():
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL, padding_side="left")
        tok.model_max_length = MAX_LEN
        ids = tok.encode(SEQS[0])
        dec = tok.decode(ids, skip_special_tokens=True).replace(" ", "")
        assert len(ids) > 1, ids
        # BPE over nucleotides: expect ~4 bp/token, and a lossless round-trip.
        assert dec == SEQS[0], f"round-trip mismatch:\n{dec}\n{SEQS[0]}"
        state["tok"] = tok
        bp_per_tok = len(SEQS[0]) / len(ids)
        return f"{len(SEQS[0])} bp -> {len(ids)} tokens ({bp_per_tok:.2f} bp/token), round-trip exact"

    @check("AutoTokenizer path used by genomeocean.llm_utils")
    def _autotok():
        # transformers>=5 can route model_type=mistral to the MistralCommon
        # backend, which has no tekken.json here. llm_utils catches that and
        # falls back; this records which branch actually fires.
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=MAX_LEN,
                                                padding_side="left", use_fast=True,
                                                trust_remote_code=True)
            tok.encode("ATGC")
            return f"AutoTokenizer worked directly -> {type(tok).__name__}"
        except Exception as e:
            return f"AutoTokenizer failed as expected ({type(e).__name__}), llm_utils falls back to PreTrainedTokenizerFast"

    # ---------------------------------------------------------- forward pass
    @check("AutoModel bf16 forward pass on GPU")
    def _fwd():
        model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to("cuda").eval()
        tok = state["tok"]
        enc = tok(SEQS, max_length=MAX_LEN, return_tensors="pt",
                  padding="longest", truncation=True)
        input_ids = enc["input_ids"].cuda()
        attn = enc["attention_mask"].cuda()
        with torch.no_grad():
            out = model.forward(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=True, return_dict=True)
        h = out.last_hidden_state
        assert h.shape[0] == len(SEQS) and h.shape[-1] == HIDDEN, h.shape
        assert torch.isfinite(h.float()).all(), "non-finite values in last_hidden_state"
        # Mean-pool exactly as llm_utils.predict does.
        m = attn.unsqueeze(-1)
        emb = (h * m).sum(1) / m.sum(1)
        assert torch.isfinite(emb.float()).all()
        # The two sequences are unrelated, so their embeddings must differ.
        cos = torch.nn.functional.cosine_similarity(emb[0].float(), emb[1].float(), dim=0).item()
        assert abs(cos) < 0.9999, f"embeddings are identical (cos={cos})"
        state["nparam"] = sum(p.numel() for p in model.parameters())
        state["emb_ref"] = emb.float().cpu().numpy()
        del model
        torch.cuda.empty_cache()
        return (f"{state['nparam'] / 1e9:.2f} B params, hidden {tuple(h.shape)}, "
                f"pooled {tuple(emb.shape)}, cos(seq0,seq1)={cos:.4f}")

    @check("bundled remote code is NOT silently used (regression guard)")
    def _remote():
        # config.json's auto_map points at a modeling_mistral.py written against
        # transformers 4.38. Under 5.x it imports but computes WRONG hidden
        # states, and crashes outright unless use_cache is disabled. The package
        # therefore pins trust_remote_code=False; this guards that decision by
        # showing the two paths genuinely disagree, so nobody "restores" it.
        tok = state["tok"]
        enc = tok(SEQS, max_length=MAX_LEN, return_tensors="pt",
                  padding="longest", truncation=True)
        ids, attn = enc["input_ids"].cuda(), enc["attention_mask"].cuda()

        crashed = None
        try:
            m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                          dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to("cuda").eval()
            with torch.no_grad():   # config.json has use_cache=true -> expected to raise
                m(input_ids=ids, attention_mask=attn, return_dict=True)
        except Exception as e:
            crashed = f"{type(e).__name__}: {str(e)[:80]}"
        else:
            del m
            torch.cuda.empty_cache()

        from genomeocean.llm_utils import LLMUtils
        import inspect
        src = inspect.getsource(LLMUtils.model.fget)
        assert "trust_remote_code=False" in src, \
            "llm_utils.LLMUtils.model no longer pins trust_remote_code=False -- " \
            "the bundled 4.38-era modeling code produces wrong embeddings on transformers 5.x"
        return (f"llm_utils pins trust_remote_code=False; remote path with the "
                f"checkpoint's own use_cache=true raises [{crashed or 'did not raise'}]")

    @check("model discriminates real DNA from random (does it actually work?)")
    def _signal():
        # The sharpest correctness check available without a labelled task: a
        # working genome LM must compress real genomic DNA below the 2 bits/bp
        # i.i.d. bound, and must do worse on shuffled/random DNA. This is what
        # catches a silently-wrong forward pass (e.g. the remote-code path,
        # which scores random DNA BETTER than real).
        seq_file = os.path.join(os.path.dirname(__file__), "..", "sample_data",
                                "dna_sequences.txt")
        with open(seq_file) as f:
            real = [l.strip() for l in f if l.strip()][:8]
        rng = np.random.default_rng(0)
        rand = ["".join(rng.choice(list("ACGT"), size=len(s))) for s in real]

        tok = state["tok"]
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        model.config.use_cache = False

        def bits_per_bp(seqs):
            nats, nbp = 0.0, 0
            for s in seqs:
                ids = tok(s, return_tensors="pt")["input_ids"].cuda()
                with torch.no_grad():
                    out = model(ids, labels=ids, use_cache=False)
                nats += out.loss.item() * (ids.shape[1] - 1)  # loss averages over n-1
                nbp += len(s)
            return nats / nbp / float(np.log(2))

        b_real, b_rand = bits_per_bp(real), bits_per_bp(rand)
        del model
        torch.cuda.empty_cache()
        assert b_real < 2.0, f"real DNA at {b_real:.3f} bits/bp does not beat the uniform bound"
        assert b_real < b_rand - 0.1, \
            f"no separation: real {b_real:.3f} vs random {b_rand:.3f} bits/bp -- forward pass is wrong"
        return (f"real DNA {b_real:.3f} bits/bp vs random {b_rand:.3f} "
                f"(uniform bound 2.000), separation {b_rand - b_real:.3f}")

    # ------------------------------------------------------------ causal LM
    @check("AutoModelForCausalLM forward + loss (perplexity path)")
    def _lm():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        tok = state["tok"]
        enc = tok(SEQS, return_tensors="pt", padding="longest")
        ids = enc["input_ids"].cuda()
        with torch.no_grad():
            out = model(ids, labels=ids)
        loss = out.loss.item()
        logits = out.logits
        assert logits.shape[-1] == VOCAB, logits.shape
        assert np.isfinite(loss), loss
        # A model that learned anything must beat the uniform baseline over a
        # 4096-token vocabulary (ln 4096 = 8.32 nats).
        uniform = float(np.log(VOCAB))
        assert loss < uniform, f"loss {loss:.3f} >= uniform baseline {uniform:.3f}"
        del model
        torch.cuda.empty_cache()
        return (f"loss {loss:.4f} nats/token (ppl {np.exp(loss):.2f}) vs uniform "
                f"{uniform:.2f}; logits {tuple(logits.shape)}")

    # -------------------------------------------------- package-level API
    @check("genomeocean.llm_utils.LLMUtils.predict (package embedding API)")
    def _pkg():
        from genomeocean.llm_utils import LLMUtils
        llm = LLMUtils(model_dir=MODEL, model_max_length=MAX_LEN)
        emb = llm.predict(SEQS, batch_size=2, do_embedding=True)
        assert emb.shape == (len(SEQS), HIDDEN), emb.shape
        assert np.isfinite(emb).all()
        # Must agree with the hand-rolled forward pass above.
        ref = state.get("emb_ref")
        extra = ""
        if ref is not None:
            d = float(np.abs(emb - ref).max())
            extra = f", max|Δ| vs manual forward = {d:.2e}"
            assert d < 0.05, f"package embedding disagrees with manual forward ({d})"
        del llm
        torch.cuda.empty_cache()
        return f"shape {emb.shape}, norms {np.linalg.norm(emb, axis=1).round(3).tolist()}{extra}"

    @check("genomeocean.embedding.llm_embed_sequences")
    def _pkg_embed():
        from genomeocean.embedding import llm_embed_sequences
        emb = llm_embed_sequences(SEQS, MODEL, batch_size=2)
        emb = np.asarray(emb)
        assert emb.shape[0] == len(SEQS) and emb.shape[1] == HIDDEN, emb.shape
        assert np.isfinite(emb).all()
        torch.cuda.empty_cache()
        return f"shape {emb.shape}"

    # ----------------------------------------------------------- generation
    if args.with_vllm:
        @check("vLLM generation via SequenceGenerator")
        def _gen():
            from genomeocean.generation import SequenceGenerator
            sg = SequenceGenerator(
                model_dir=MODEL, prompts=[SEQS[0][:40]], num=2,
                min_seq_len=16, max_seq_len=64, temperature=1.0, top_p=0.9,
                seed=123, gpu_memory_utilization=args.gpu_mem_util,
            )
            df = sg.generate_sequences(prepend_prompt_to_output=False)
            assert len(df) == 2, len(df)
            seqs = df["seq"].tolist()
            for s in seqs:
                assert len(s) > 0, "empty generation"
                assert set(s) <= set("ACGTN"), f"non-nucleotide output: {sorted(set(s))}"
                assert "N" not in s, "N was generated despite logit_bias"
            return f"{len(seqs)} seqs, lengths {[len(s) for s in seqs]}, first: {seqs[0][:60]}"
    else:
        print("\n(skipping vLLM generation; pass --with-vllm to include it)")

    # -------------------------------------------------------------- summary
    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nfail}/{len(RESULTS)} checks passed")
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
