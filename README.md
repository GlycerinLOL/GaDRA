# GaDRA

A HuggingFace **`peft`-native** implementation of **GaDRA** (Gated Dual-conditioned Residual Adapter):
a parameter-efficient adapter with a per-token, input-conditioned residual gate that learns *when not to
apply* the LoRA update. Model-agnostic — works on any HF `*ForCausalLM` via standard `target_modules`
name-matching. This is the official open-source implementation of the GaDRA paper.

## Why this repo exists

The original research implementation hard-bound GaDRA to Llama by reimplementing the whole decoder stack
(~2600 lines) just to thread per-token gate values through the forward pass. This package re-expresses GaDRA
as a `BaseTuner` / `BaseTunerLayer` custom method registered via `peft.utils.register_peft_method`, so you
call it through the normal peft API — swap `LoraConfig` for `GaDRAConfig` and nothing else changes (except
that GaDRA is non-mergeable).

## Install

```bash
pip install gadra        # pulls peft==0.16.0, transformers==4.53.3
```

Importing the package registers the `gadra` method with peft (adds `PeftType.GADRA`). This is the method
only — zero data/eval/inference dependencies. To **reproduce the paper** from a checkout, use uv (below).

## Reproduce the paper (uv)

The repo's training + inference workflow is managed with [uv](https://docs.astral.sh/uv/). One environment
covers single-GPU and multi-GPU (they install the same packages; multi-GPU only changes the launch).

**Prerequisites** (uv cannot install these — they are host-level):
- NVIDIA GPU with **driver ≥ 525.60.13** (the env uses the CUDA 12.4 runtime, bundled in the torch wheel).
- Linux x86_64, glibc ≥ 2.17 (uv-managed CPython 3.12 + all wheels are manylinux2014).
- ~25 GB disk for the env + cached wheels.
- A Hugging Face account with access to the **gated** base model: request access to
  `meta-llama/Llama-3.1-8B-Instruct`, then `huggingface-cli login` (or set `HF_TOKEN`).

### 1. Environment

```bash
# install uv (one-time; no conda, no system python — uv fetches CPython 3.12 per .python-version)
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone + create the GPU env (cu124 torch + prebuilt flash-attn + deepspeed; no compilation)
git clone <repo-url> GaDRA && cd GaDRA
uv sync --group gpu
huggingface-cli login                          # for the gated Llama base model
```

### 2. Data (not shipped — you provide it)

The run-configs point at `data/*.jsonl` placeholders; supply your own files and edit the config (or
`--override train_file=...`). Formats:
- **Training corpus** (`train_file`): JSONL, one `{"text": "..."}` per line (continual-pretraining text).
  The paper uses BBC-News / CC news corpora — see the paper for sourcing; any in-domain text in this format works.
- **Eval sets** (`eval_file`): JSONL — `task: qa` → `{"prompt": "...", "answer": "..."}`;
  `task: gsm8k` → `{"prompt": "...", "answers": ["..."]}`. GSM8K / MBPP are public; format them to this schema.

### 3. Train

```bash
# GaDRA (default: dual gate, hard/Gumbel) — single GPU
uv run python -m examples.train --config examples/config/train.yaml
#   variants, one override each:  --override router_conditioning=mono   (GaDRA-Mono)
#                                 --override gate=soft                   (GaDRA-Soft)

# LoRA baseline (stock peft — the method GaDRA is compared against)
uv run python -m examples.train_lora_baseline --config examples/config/train_lora.yaml

# multi-GPU (any of the above) = the paper's workflow: accelerate launch + DeepSpeed ZeRO-2
uv run accelerate launch --num_processes <N_GPUS> \
    --config_file examples/config/deepspeed_zero2.yaml \
    -m examples.train --config examples/config/train.yaml
```

### 4. Inference / eval

```bash
# deterministic (no key): task: qa | gsm8k | mbpp
uv run python -m examples.inference --config examples/config/inference.yaml

# GPT-judged "Correct %": task: bbcqa | tiebe  (key from the environment — never the repo)
export OPENAI_API_KEY=sk-...
uv run python -m examples.inference --config examples/config/inference.yaml --override task=bbcqa
```

The eval derives the prompt / reference / judge inputs from each sample the same way the research harness
does (prompt from `text`/`messages`/`question`, reference from `messages[-1]`, question from `messages[-2]`,
BBC-QA judge document from a `documents_file` keyed by `id`), so the paper's **raw** eval files work directly.
Same chat template + greedy params ⇒ the **deterministic** numbers (QA char-F1/EM, GSM8K EM, MBPP pass@1)
reproduce bit-for-bit. **BBC-QA / TiEBe** use the verbatim GPT judge (`bbcqa` / `tiebe`) for the paper's
"Correct %" — these need `OPENAI_API_KEY` in the environment (BBC-QA also needs `documents_file:` pointing at
the source-article JSONL) and are judge-dependent (not bit-exact, since they call OpenAI). No key is ever
stored in the repo. (MBPP needs `HF_ALLOW_CODE_EVAL=1`.)

### 5. SLURM (offline compute nodes)

Set `GADRA_REPO` to your shared-filesystem checkout, pre-sync once on the login node
(`uv sync --group gpu --locked`), then submit — [`examples/slurm/train.slurm`](examples/slurm/train.slurm)
(multi-GPU train, accelerate + ZeRO-2) and [`examples/slurm/inference.slurm`](examples/slurm/inference.slurm)
(single-GPU eval). Both handle the offline/uv flags; the inference wrapper reads `OPENAI_API_KEY` from the
submitting environment for `bbcqa`/`tiebe` (never from the repo). On an account/partition cluster, set
`export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>` once (Slurm reads them — no file edit).

The default packing (`fa2_collator`) is paper-faithful and needs flash-attention-2 (installed by the `gpu`
group as a prebuilt wheel — no build). For a machine without the matching wheel, use the portable SDPA path:
`--override packing=group` (not bit-exact to the paper). See [`examples/README.md`](examples/README.md).

## Usage

### Train (standard `transformers.Trainer`, plain LM loss)

```python
import gadra                              # side-effect: registers the "gadra" method
from gadra import GaDRAConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, Trainer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
cfg = GaDRAConfig(
    r=512, lora_alpha=1, lora_dropout=0.05,
    target_modules=["up_proj", "gate_proj", "down_proj"],
    router_conditioning="dual",   # "dual" = GaDRA | "mono" = GaDRA-Mono
    gate="hard",                  # "hard" = Gumbel-STE | "soft"
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
Trainer(model=model, args=..., train_dataset=...).train()   # no custom trainer, no aux loss
model.save_pretrained("out/")
```

A config-driven runnable script with the paper's §A.1 recipe is [`examples/train.py`](examples/train.py) —
run it as `uv run python -m examples.train --config examples/config/train.yaml` from a repo checkout
(repo-only tooling; see [Reproduce the paper](#reproduce-the-paper-uv) and [Scope](#scope)).

### Generate

```python
import gadra
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "out/")     # gate stays attached (non-mergeable)
model.generate(...)
```

For quick interactive generation, use the standard peft API (above). For paper-repro **evaluation**, see
[`examples/inference.py`](examples/inference.py) and [Reproduce → Inference](#4-inference--eval).

### Variants

| Variant | `router_conditioning` | `gate` |
|---|---|---|
| GaDRA | `dual` | `hard` |
| GaDRA-Mono | `mono` | `hard` |
| GaDRA-Soft | `dual` | `soft` |
| GaDRA-Mono-Soft | `mono` | `soft` |

### Non-mergeable

GaDRA's gate is per-token and input-dependent, so the adapter cannot be folded into the base weights.
`merge_and_unload()` / `merge_adapter()` raise a clear `NotImplementedError` by design — keep the adapter
attached at inference.

## Convert a legacy checkpoint (repo tooling)

Checkpoints from the original research code (`peft_config.json` + `peft_model.bin`) convert to the peft
format with the repo-only converter under `examples/` (not part of the pip package — clone the repo and
`uv sync --group gpu`):

```python
from examples.convert import convert_checkpoint   # run from a repo checkout, not `pip install gadra`
convert_checkpoint("old_ckpt/", "out/")   # writes adapter_config.json + adapter_model.safetensors
```

## Method (paper)

Output `y = y⁰ + γ·δ`, with `δ = α·B·A·x` and a per-module affine gate producing the per-token scalar `γ`:
the **dual** gate conditions on `[y⁰; δ]` (GaDRA), the **mono** gate on `δ` only. `γ` is binary via a
Gumbel-sigmoid straight-through estimator during training (deterministic `1[σ(z) > 0.5]` at inference), or a
continuous `soft` gate. The default config (`r=512, α=1`, on `up_proj`/`gate_proj`/`down_proj`) is paper-faithful.

## Environment

Pinned to `peft==0.16.0`, `transformers==4.53.3`, Python 3.12. See [`docs/DESIGN.md`](docs/DESIGN.md) for the
full architecture mapping, phased plan, and numeric-parity protocol.

## Scope

**The pip package (`pip install gadra`) is the method only** — the GaDRA tuner + the standard peft/Trainer
training path, with zero data / eval / inference dependencies. It is a clean, model-agnostic `peft` variant
you can drop into any project.

**Reproducing the paper** uses the repo-only tooling under [`examples/`](examples/) (never shipped in the
wheel): `examples/processing.py` (the parity-exact FA2 packing / EOS-append / tokenizer that fed the paper's
CPT), `examples/evaluation.py` (the verbatim BBC-QA / GSM8K / MBPP / TiEBe scorers + greedy runner), and
`examples/convert.py` (the legacy-checkpoint converter), driven by `examples/train.py` / `examples/inference.py`
+ `examples/config/`. Clone the repo and `uv sync --group gpu` (see [Reproduce the paper](#reproduce-the-paper-uv)).
See [`examples/README.md`](examples/README.md).

**Out of scope:** other PEFT methods (MiLoRA / LoRA-Null / CLoRA) have upstream implementations; and the
paper's per-token **analysis tooling** (contribution-ratio / γ extraction) stays in the research repo.

## Security

The optional MBPP eval executes model-generated code (`HF_ALLOW_CODE_EVAL=1`) — run it only in an isolated
environment. No secrets are stored in the repo; the GPT judge reads `OPENAI_API_KEY` from the environment. See
[`SECURITY.md`](SECURITY.md) for the disclosure policy.

## License

Licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use GaDRA, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button). The accompanying paper citation will be added upon publication.
