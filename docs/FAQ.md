# FAQ / Troubleshooting

## Installation

**`pip install gadra` doesn't install the training / eval tools.**
By design — the pip package is the *method only* (zero data / eval / inference deps). To reproduce the paper,
clone the repo and `uv sync --group gpu`. See the README → *Reproduce the paper*.

**flash-attn fails to build / install.**
Use uv (`uv sync --group gpu`) — it installs a *prebuilt* flash-attn wheel (cp312 / torch 2.6 / cu12), no
compilation. If you can't use FA2, train with `--override packing=group` (portable SDPA path; not bit-exact
to the paper).

**CUDA / driver errors at import.**
The env uses the CUDA 12.4 runtime bundled in the torch wheel; you only need an NVIDIA driver ≥ 525.60.13.
uv cannot install the driver — that is the one host-level prerequisite.

**`meta-llama/Llama-3.1-8B-Instruct` won't download.**
It is a gated model — request access on the Hub, then `huggingface-cli login` (or set `HF_TOKEN`).

## Running

**SLURM job can't reach the network / model.**
Compute nodes are usually offline. Pre-sync on the login node (`uv sync --group gpu --locked`) and pre-cache
the model; the wrappers set `HF_HUB_OFFLINE=1`. On an account/partition cluster, set
`export SBATCH_ACCOUNT=... SBATCH_PARTITION=...` once (Slurm reads them — no file edit).

**MBPP eval errors about code execution / does nothing.**
MBPP pass@1 executes model-generated code; set `HF_ALLOW_CODE_EVAL=1` and run it **only in an isolated /
containerized environment** (see [`SECURITY.md`](../SECURITY.md)).

**BBC-QA / TiEBe "Correct %" needs a key.**
Those use a GPT judge — set `OPENAI_API_KEY` in your environment (never commit it). The deterministic tasks
(QA / GSM8K / MBPP) need no key.

**Can I merge the adapter into the base weights?**
No. GaDRA's gate is per-token and input-dependent, so it is non-mergeable by design — keep the adapter
attached at inference. `merge_and_unload()` / `merge_adapter()` raise a clear error.

## Reproducing

**Which numbers reproduce bit-for-bit?**
The deterministic metrics (QA char-F1/EM, GSM8K EM, MBPP pass@1) reproduce exactly given the same chat
template + greedy params. BBC-QA / TiEBe "Correct %" use the GPT judge and are method-equivalent, not
bit-exact (they depend on the OpenAI model version).

**Data isn't included.**
Correct — supply your own JSONL (see the README → *Data* for formats). The paper uses BBC-News / CC news
corpora; any in-domain text in the documented format works.
