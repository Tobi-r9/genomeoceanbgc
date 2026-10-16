# GenomeOcean / BGC foundation model on JUPITER — agent notes

Fork of [jgi-genomeocean/genomeocean](https://github.com/jgi-genomeocean/genomeocean)
(origin here is `Tobi-r9/genomeoceanbgc`), set up on **JUPITER** (JSC, aarch64 /
GH200) to work with **`pGenomeOcean/GenomeOcean-4B-bgcFM`** — GenomeOcean-4B
finetuned on 11 M biosynthetic gene cluster sequences.

Upstream targets x86 + NERSC Perlmutter (A100) and ships Docker/Apptainer images
built on `nvcr.io/nvidia/vllm`. **Neither container is used here** — JUPITER only
allows fakeroot Apptainer builds and the NGC images are x86-oriented, so this is
a native venv install. Everything JUPITER-specific lives in [jupiter/](jupiter/);
the environment lives outside the repo in `/e/project1/scifi/thoeppe/envs/bgc/`.

**Status: environment built, model staged, all 10 smoke-test checks pass —
embedding, forward pass, perplexity and vLLM generation. See "Verified" below for
what was measured, and "Three things that were wrong out of the box" for the
non-obvious fixes that got there.**

---

## Quick start

```bash
# one-time, on a LOGIN node (compute nodes have no outbound internet)
bash /e/project1/scifi/thoeppe/envs/bgc/setup.sh              # ~13 min, ~8 GB
bash /e/project1/scifi/thoeppe/genomeoceanbgc/jupiter/download_model.sh   # 8 GB

# every session
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
cd /e/project1/scifi/thoeppe/genomeoceanbgc

# verify the stack (login node has a GH200; the transformers checks take ~30 s)
python jupiter/test_bgc_model.py
python jupiter/test_bgc_model.py --with-vllm     # + generation, ~25 GB HBM
sbatch jupiter/test_model.sbatch                 # same, on a booster node
```

`$GO_BGC_MODEL` is exported by `activate.sh` and points at the staged model, so
pass that rather than the repo id — compute nodes cannot reach huggingface.co.

```python
from genomeocean.llm_utils import LLMUtils
import os
llm = LLMUtils(model_dir=os.environ["GO_BGC_MODEL"], model_max_length=10240)
emb = llm.predict(sequences, batch_size=25)      # (N, 3072) mean-pooled
```

## Layout

| Path | What |
|---|---|
| `/e/project1/scifi/thoeppe/envs/bgc/` | env: `setup.sh`, `modules.sh`, `config.sh`, `activate.sh`, `requirements.txt`, `venv` → scratch |
| [jupiter/download_model.sh](jupiter/download_model.sh) | pre-stages checkpoints (`bgcFM`, `4B`, `500M`, `100M`, `detector`) |
| [jupiter/test_bgc_model.py](jupiter/test_bgc_model.py) | the smoke test; 9 independent PASS/FAIL checks + optional vLLM |
| [jupiter/test_model.sbatch](jupiter/test_model.sbatch) | same test on a booster node |
| [genomeocean/](genomeocean/) | upstream library (2 files patched, see below) |
| [examples/](examples/), [go_generate.py](go_generate.py), [go_scan.py](go_scan.py) | upstream CLIs — all default to `pGenomeOcean/GenomeOcean-4B`, override `--model_dir` |
| [docker/](docker/), [apptainer/](apptainer/) | upstream container recipes — **reference only, not used on JUPITER** |
| [benchmarks/](benchmarks/) | upstream's A100 vLLM V0→V1 report — **not JUPITER numbers** |
| `/e/scratch/profound/thoeppe/genomeocean/` | `envs/bgc-venv/`, `hf_models/`, `hf_cache/`, `logs/`, `outputs/`, JIT caches |

## Environment

Built as a **standalone venv, everything from PyPI** — deliberately *unlike*
`envs/spg/` and `envs/svgd/`, which layer on EasyBuild's `PyTorch` module via
`--system-site-packages`. Reason: **vLLM pins torch exactly** (`vllm==0.26.0` →
`torch==2.11.0`), so a module-provided torch cannot be used. Do not add
`module load PyTorch` to `modules.sh`.

- `Stages/2025` → **Python 3.12.3**. Stages/2026 only offers Python 3.13, and
  upstream tests 3.10–3.12. Stages/2025 is deprecated but still loadable.
- Installed and verified: **torch 2.11.0+cu130**, **vllm 0.26.0**,
  **transformers 5.14.1**, tokenizers 0.22.2, peft 0.20.0, accelerate 1.14.0,
  numpy 2.3.5, pandas 3.0.5, scikit-learn 1.9.0, biopython 1.87,
  pyrodigal 3.7.1, bitsandbytes 0.50.0, genomeocean 0.6.0 (editable).
- All aarch64 CUDA wheels exist for this combination; nothing is built from
  source. The pip `nvidia-*-cu13` wheels supply CUDA 13.0; the driver reports
  CUDA 13.2, which is compatible.

### The venv lives on scratch, and that is not optional

`envs/bgc/venv` is a **symlink** to
`/e/scratch/profound/thoeppe/genomeocean/envs/bgc-venv` (8.1 GB, ~82 k files).

`/e/project1/scifi` was, at setup time, **over its data soft quota** (21.8 of
20 TiB, hard limit 22 TiB) and at **~96 % of its 4 M inode soft limit**
(3.84 M used). An 82 k-file venv there would have pushed inodes to the limit and
broken writes for the whole `scifi` project. Check before adding anything large:

```bash
jutil project dataquota -p scifi      # and -p profound
```

`profound` scratch has room (85 of 200 TiB, 6.2 M of 8 M inodes).

## Three things that were wrong out of the box

### 1. `trust_remote_code=True` silently produces garbage on transformers 5.x

**This is the important one.** The checkpoints carry their own
`modeling_mistral.py` + `configuration_mistral.py` (`auto_map` in
`config.json`), written against **transformers 4.38.2**. Under transformers 5.x
that code still *imports*, so nothing looks broken — but it computes wrong
hidden states:

Measured over all 20 × 2000 bp sequences in `sample_data/dna_sequences.txt`, and
against length-matched uniform-random ACGT as a control:

| | real DNA | random DNA | verdict |
|---|---|---|---|
| native transformers Mistral | **6.091 nats/tok** (1.75 bits/bp) | 6.827 (2.04 bits/bp) | compresses real DNA, as it must |
| bundled remote code | 7.236 nats/tok | **7.173 nats/tok** | scores *random* DNA better — no signal at all |

Embeddings differ by up to **3.4 absolute on a ±8 scale** (mean-pooled vector
norms 18.1 vs 28.4). Both paths are internally self-consistent and
padding-invariant, so no error surfaces — you just get meaningless numbers.

It also **crashes outright** unless `use_cache` is disabled, because
`DynamicCache.from_legacy_cache` was removed in transformers 5:
`AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'`.
`config.json` sets `use_cache: true`, so a plain
`AutoModel.from_pretrained(..., trust_remote_code=True)` followed by a forward
pass raises. `LLMUtils.model` happened to set `config.use_cache = False`, which
suppressed the crash and left only the silent wrongness.

**Patched** `genomeocean/llm_utils.py` and `genomeocean/embedding.py` to
`trust_remote_code=False` (and `torch_dtype=` → `dtype=`, deprecated in
transformers 5). Nothing is lost: the checkpoint is plain Mistral, confirmed
against the weight shapes — `q_proj [3072,3072]`, `k_proj/v_proj [1024,3072]`
→ 12 heads × 256 head_dim, 4 KV heads — and transformers 5.14 correctly migrates
`rope_theta: 1e6` into `rope_parameters`. `sliding_window` is null.

`compute_sequence_perplexity` / `compute_token_perplexity` already used
`trust_remote_code=False`, so they were always on the correct path.

`jupiter/test_bgc_model.py` guards this two ways: a real-vs-random bits/bp
assertion, and a check that `LLMUtils.model` still pins
`trust_remote_code=False`. **Do not "restore" `trust_remote_code=True`.**

### 2. vLLM's FlashInfer JIT needs `CUDA_HOME` **and** a self-consistent nvcc

vLLM's default sampler on this platform is FlashInfer, which nvcc-compiles
kernels at first use. Getting that to work took two fixes, both in the env.

**2a — no `/usr/local/cuda` on JUPITER.** Engine startup died with:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

`config.sh` now exports `CUDA_HOME` to the **pip** CUDA 13 toolkit that arrives
with `torch+cu130` (`$ENV_DIR/lib/python3.12/site-packages/nvidia/cu13` — a
complete toolkit: `bin/nvcc`, `include/`, `lib/`, `nvvm/`) and prepends its `bin`
to `PATH`. Deliberately *not* `module load CUDA`, which could be a different
major version than the cu130 wheels.

**2b — vLLM's own dependency set ships a skewed CUDA toolchain.** With
`CUDA_HOME` set it got further and then failed to compile, because
`vllm==0.26.0` resolves `nvidia-cuda-runtime` to **13.0**.96 (so
`CUDART_VERSION 13000`) but `nvidia-cuda-nvcc` / `-crt` / `nvidia-nvvm` to
**13.3**.73. FlashInfer's bundled CCCL requires an exact major.minor match:

```
error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

and once the headers are satisfied, `ptxas` from nvcc 13.0 rejects PTX emitted by
nvvm 13.3:

```
ptxas fatal : Unsupported .version 9.3; current version is '9.0'
```

`requirements.txt` therefore pins `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and
`nvidia-nvvm` to **13.0.88**, matching the runtime headers. Only the JIT
toolchain is affected — nothing links against nvcc and torch keeps its cu130
runtime. `setup.sh` asserts `nvcc` and `CUDART_VERSION` still agree and fails the
build if they drift, so a future `vllm` bump surfaces this in seconds rather than
70 s into an engine startup.

**2c — the pip CUDA wheels have no linker symlinks.** They are a *runtime*
distribution: `lib/libcudart.so.13` exists, but the unversioned `libcudart.so`
does not, and there is no `lib64/` at all. FlashInfer links with
`-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda`:

```
ld: cannot find -lcudart: No such file or directory
```

`setup.sh` creates the two missing links inside the venv (`lib64 → lib` and
`libcudart.so → libcudart.so.13`). `-lcuda` needs nothing extra — the real driver
is at `/usr/lib64/libcuda.so`, which `ld` searches by default, so no stub is
required.

`FLASHINFER_CACHE_DIR` is also redirected to scratch — it defaults to
`$HOME/.cache/flashinfer` (small quota), and on scratch a login node's compiled
kernels are reused by compute-node jobs.

### 3. `llm_embed_sequences` silently truncated local models to 1024 tokens

`MODEL_TOKEN_LIMITS` in `genomeocean/embedding.py` is keyed by **repo id**, and
anything not found that happens to be a directory fell back to
`model_max_length = 1024` with only a `logger.warning`. Since compute nodes
force local paths, every call was silently capped at 1/10th of the 4B model's
10240 tokens — quietly discarding most of every long sequence.

**Patched**: added `GenomeOcean-4B-bgcFM` and the artificial detector to
`MODEL_TOKEN_LIMITS`, plus `_infer_token_limit()`, which matches a local
directory to a known checkpoint by basename. Keep the staged directory named
after the checkpoint (`hf_models/GenomeOcean-4B-bgcFM`) or the 1024 fallback
returns.

## Verified

`python jupiter/test_bgc_model.py --with-vllm` — **10/10 checks pass** on one
GH200, peak GPU memory **8.7 GiB** for the transformers checks. Confirmed twice:
on a login node, and on a booster node fully offline
(`sbatch jupiter/test_model.sbatch`, job **1247855**, COMPLETED in 7:42).

| Check | Result |
|---|---|
| config, native mistral | 24 layers, hidden 3072, 12 heads / 4 KV, vocab 4096 |
| tokenizer round-trip | 81 bp → 21 tokens, exact round-trip |
| `AutoTokenizer` | works directly (`TokenizersBackend`) — the MistralCommon fallback never fires here |
| `AutoModel` bf16 forward | 4.24 B params, `last_hidden_state (2, 21, 3072)`, pooled `(2, 3072)` |
| remote-code regression guard | native path pinned; remote path raises as expected |
| **real vs random DNA** | **1.704 vs 2.040 bits/bp** (uniform bound 2.000) |
| `AutoModelForCausalLM` + loss | 6.272 nats/token, logits `(2, 21, 4096)` |
| `LLMUtils.predict` | `(2, 3072)`, agrees with the manual forward to 3.1e-02 (bf16 noise) |
| `llm_embed_sequences` | `(2, 3072)` |
| **vLLM generation** (`SequenceGenerator`) | 2 sequences, 262 and 252 bp, pure ACGT, no `N` |

The **bits/bp** check is the one that matters: a working genome LM must compress
real genomic DNA below the 2 bits/bp i.i.d. bound and must do worse on random
DNA. It is what caught finding #1. The test uses the first 8 sequences of
`sample_data/dna_sequences.txt` to stay fast (1.704 bits/bp); over all 20 it is
1.755. These are generic 2000 bp sequences, **not** BGCs — a BGC-specific
benchmark would be a fairer test of this checkpoint and has not been run.

bf16 vs fp32 embeddings differ by 0.047 max absolute; bf16 is fine.

## Model facts

`GenomeOcean-4B-bgcFM`, 8.0 GB, 2 safetensors shards, 4.24 B params, bf16.

- **Mistral architecture**, vocab 4096, `max_position_embeddings` 32768 but the
  **usable limit is 10240 tokens**; embedding dim **3072**.
- Tokenizer is **BPE over nucleotides**, `PreTrainedTokenizerFast`, ~**5 bp per
  token** on real DNA (3.86 on the short GC-rich test strings). So 10240 tokens
  ≈ **51 kbp**. `[CLS]`=1 is bos, `[SEP]`=2 eos, `[PAD]`=3, `[MASK]`=4.
- `lm_head.weight` is reported `UNEXPECTED` when loading with `AutoModel` —
  harmless, `AutoModel` has no LM head. `tie_word_embeddings` is false.
- `llm_utils.generate` prepends the literal string `"[CLS]"` to every prompt and
  suppresses token 8 (`N`) via `logit_bias`. Sequences are mean-pooled over the
  attention mask for embeddings.
- Set `model_max_length` explicitly to 10240 (`LLMUtils` defaults to it; the
  example scripts assert `<= 10240`).

## Cluster facts

- Partition `booster`, account `scifi`, `--gres=gpu:4`, nodes allocated
  **exclusively** (288 CPUs, 858 GB RAM). Max walltime **12 h**.
- **Login nodes have a GH200 and internet** — used here for setup, download, and
  the whole smoke test. Keep login-node GPU work to quick checks.
- Compute nodes have **no outbound internet**. `test_model.sbatch` sets
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so anything unstaged fails
  loudly instead of hanging on retries.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is set in `config.sh` (fork is unsafe once
  CUDA is initialised). **Never set `VLLM_USE_V1`** — removed in vLLM ≥ 0.11,
  setting it errors. Upstream's README says the same.
- `LLMUtils` picks `tensor_parallel_size` from `torch.cuda.device_count()`,
  rounded down to a divisor of **12** (the attention-head count) by
  `max_divisor_of_12`. On a 4-GPU booster node that gives TP=4; on 8 GPUs it
  would give 6, not 8. Set `CUDA_VISIBLE_DEVICES` to control it.
- `llm_utils.py` sets `enforce_eager=True` (CUDA graphs off) with a TODO to
  re-test — upstream saw capture hangs on aarch64/GB10. Not re-tested here.

### Harmless noise in the logs

- `The activation script must be sourced…` — `activate.sh`'s `$0` check
  misfiring when sourced from a script (same in `envs/spg/`, `envs/svgd/`).
- `MistralModel LOAD REPORT … lm_head.weight UNEXPECTED` — see above.
- `` `torch_dtype` is deprecated! Use `dtype` instead `` and
  `` `use_return_dict` is deprecated `` — from upstream call sites not yet
  migrated (the two patched ones are).
- `AttentionMaskConverter … deprecated and will be removed in Transformers
  v5.10` — only fires on the remote-code path, which is no longer used.
- `Ignoring clean_up_tokenization_spaces=True for BPE tokenizer` — from the
  checkpoint's `tokenizer_config.json`; decoding is unaffected because
  `llm_utils.generate` strips spaces itself.
- `No FlashInfer autotune cache entries found. Falling back to default tactics.`
- `Stages/2025 is deprecated`.

## vLLM generation — works, and what to expect

Verified end to end at `gpu_memory_utilization=0.25` on one GH200: engine init
20.2 s (weights 7.94 GiB in 3.4 s, KV cache 13.78 GiB = 150,448 tokens), whole
check 38.6 s. Getting there needed all three parts of finding #2.

**The first run on a fresh cache is much slower** — FlashInfer nvcc-compiles its
sampling kernels (~2 min, ~6 MB). The cache lives on scratch at
`$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>/` and is already warm
for `sm_90a`, so compute-node jobs reuse what a login node built (verified: a
generation run off the scratch cache does not recompile). If you see a
~2-minute stall at engine startup, that is a cold JIT cache, not a hang.

> The knob is **`FLASHINFER_WORKSPACE_BASE`** — a *base* directory to which
> flashinfer appends `.cache/flashinfer` (`flashinfer/jit/env.py:51`).
> `FLASHINFER_CACHE_DIR` looks like the right name and is what you will reach for
> first, but it is a module constant that is **not** read from the environment.

**Engine init can fail transiently on a login node.** `RuntimeError: Engine core
initialization failed` with nothing obviously wrong in the traceback usually
means another process is holding GPU memory — login nodes are shared, and
`gpu_memory_utilization` is a fraction of the *total* 97 GB, not of what is free.
Check `nvidia-smi` and retry, or lower `--gpu-mem-util`, before assuming a real
break.

`enforce_eager=True` in `llm_utils.py` means **no CUDA graphs and no
torch.compile** ("Enforce eager set, disabling torch.compile and CUDAGraphs").
Upstream set this for an aarch64/GB10 capture hang and left a TODO to re-test.
Flipping it to `False` is the obvious first thing to try if generation throughput
matters — untested here.

## Not done yet

- No BGC-specific evaluation. The bits/bp number uses generic sample sequences;
  this checkpoint was finetuned on BGCs and should be judged on those.
- Only the `bgcFM` checkpoint is staged. `download_model.sh 4B 100M` fetches
  others if a base-model comparison is wanted.
- `unittests.py` and `tests/test_capabilities.py` are hardcoded to
  `pGenomeOcean/GenomeOcean-100M` and need internet (or that model staged);
  `tests/` also expects an embedding dim of 768. Not run.
- Multi-GPU / DDP embedding (`embedding.py` has a `DistributedSampler` path) is
  untested here.
