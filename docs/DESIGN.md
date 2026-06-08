# GaDRA-PEFT — Design & Architecture

> How the GaDRA paper's method maps to a HuggingFace `peft`-native custom tuner (`BaseTuner` /
> `BaseTunerLayer`), and the numeric-parity protocol against the original research implementation.
> Basis: the GaDRA paper *"Learning When Not to Apply LoRA in Replay-Free Continual Pre-Training"*
> (method §3–4, setup §5 / §A.1). Pinned env (paper §A.2): PyTorch 2.6.0, Transformers 4.53.3,
> peft 0.16.0, Python 3.12.

## 0. Purpose & scope

Ship the **GaDRA method + its training** as a HuggingFace `peft`-native custom tuner — model-agnostic (any HF
`*ForCausalLM` via `target_modules` name-matching), pip-installable, used through the standard peft API
(`get_peft_model`, `save_pretrained`, `from_pretrained`) and the **standard `transformers.Trainer`**.

**In scope = exactly the paper's method (no more, no less):**

| Element | Spec | Paper cite |
|---|---|---|
| Adapter | LoRA, **r=512, α=1**, on **up_proj / gate_proj / down_proj** of every decoder block | §5.2 l.373-381; §A.1 l.977-981 |
| Gate | one **affine router per module** `g = wᵀ[·] + b_g`, **b_g init positive** (gates start open) | §4 l.305-309 |
| Conditioning | **dual** `g([y⁰;δ])` (GaDRA) and **adapter-only** `g(δ)` (GaDRA-Mono) | §4 Eq.3 l.261 |
| Gate type | **hard** binary via **Gumbel-sigmoid STE**, τ=1 no-anneal; **inference deterministic** `1[σ(z)>0.5]`; **soft** `γ∈[0,1]` ablation | §4 l.291-303; Tab.6 |
| Output | `y = y⁰ + γ·δ`, `δ = α·B·A·x`; non-mergeable (γ per-token, input-dependent) | §4 Eq.2 |
| Training | standard `Trainer`, plain LM loss; AdamW, cosine LR, **BF16**, **DeepSpeed ZeRO-2**, FA2, seqlen 1024, gbs 128/mb 16, clip 0.1, wd 0.01, **seed 42, 3 epochs**; per-cell LR/warmup (Tab.8) | §A.1 l.969-986 |
| Backbones | Llama-3.1-8B-Instruct (primary), Qwen3-8B (probe) — same MLP-projection adapters → name-matching covers both (and any HF model) | §5.2 |
| Variants shipped | GaDRA (dual+hard), GaDRA-Mono (mono+hard), GaDRA-Soft (dual+soft), Mono-Soft | §6 / Tab.6 |

**Reproduction tooling — repo-only `examples/`, NOT in the wheel (package/repo split 2026-06-04 §0.3; uv + flattening 2026-06-05 §0.4):**
> The data + eval code below lives under the repo-only, flattened `examples/` tree and is **not** shipped by
> `pip install gadra`. The pip package is the method only; clone the repo + `uv sync --group gpu` to
> reproduce the paper (`python -m examples.train` / `python -m examples.inference`, config-driven).
- **`examples.processing`** (was `gadra.data`) — the parity-exact data path that fed the paper's CPT: FA2 `PackingCollator` (varlen,
  no `attention_mask`, per-doc `range()` position_ids), PT tokenize (`add_special_tokens=True`,
  truncation@max_len, `labels=copy`), EOS-append + auto-detection, tokenizer build, plus a generic
  `group_texts` strategy for broad reuse. Ported verbatim from the research repo and golden-tested (G2).
- **`examples.evaluation`** (was `gadra.eval`) — verbatim BBC-QA / GSM8K / MBPP / TiEBe scorers + greedy runner that
  reproduce the paper's deterministic numbers, plus a PPL harness. (Generic-benchmark delegation to lm-eval is a future stub.)

**Out of scope (stays in research repo):**
- **MoE / whole-MLP `mlp_external`** — paper lists MoE as **future work** (§7 l.601-605; Limitations l.668-677). No MLPWrapper.
- Attention-only reimplementation; `router_type="Add"`; `router_input ∈ {adapter_input, down_proj}`;
  `gamma_hard_masking="STE"` (noise-free); `init_method="svd_minor"` (MiLoRA); `peft_type="adapter"` (relu/tanh);
  manual partial `gate_layers`/`peft_layers` DSL (the per-(layer×module) budget is **learned**, not set).
- **Per-token analysis tooling** (the paper's diagnostics): CR/γ/R2 extraction (`gamma_values.jsonl`/
  `cr_values.jsonl`), `gamma_override` interventions (§6.2), `.generate()` gamma collection, regularizer
  losses (gamma-target / CR / weight — off in every config), forgetting-loss, score notebooks. Research repo only.
- The **broad research eval harness** (4238-line `inference_common`, 13-task evaluator dispatch, GPT/NLI
  judges, MT-Bench) — `examples.evaluation` ports only the 4 canonical-task scorers, not the harness.
- **No auxiliary training loss → no per-layer collection, no stash sidecar, no custom Trainer, no reentrant concern.**

**Note (generic tuner):** name-matching wraps any `nn.Linear`, so targeting attention is *possible* for users at no
extra code; the documented/default GaDRA config targets the three MLP projections (paper-faithful). We keep the tuner
generic (the natural peft behavior) rather than adding code to forbid non-MLP targets.

## 1. Feasibility — FEASIBLE (clean, pure adapter)

`PeftBlock.forward` is a pure function of `(x, base_linear(x))` (`peft_model.py:793-831`); a single-`nn.Linear`
`BaseTunerLayer` wrapper reproduces it, and name-matching injection replaces the ~2600-line Llama rewrite. Non-mergeable
is fine (`BaseTunerLayer.merge/unmerge` default to `NotImplementedError`; `get_peft_model`/save/load never merge).

## 2. User-facing API

```python
pip install gadra
from gadra import GaDRAConfig
from peft import get_peft_model, PeftModel
from transformers import Trainer

cfg = GaDRAConfig(r=512, lora_alpha=512, lora_dropout=0.05,   # lora_alpha=r => peft scaling alpha/r = 1.0
                  target_modules=["up_proj", "gate_proj", "down_proj"],
                  router_conditioning="dual",   # "dual" (GaDRA) | "mono" (GaDRA-Mono)
                  gate="hard",                   # "hard" (Gumbel-STE) | "soft"
                  gamma_threshold=0.5, tau=1.0)
model = get_peft_model(base_model, cfg)
Trainer(model=model, args=..., train_dataset=...).train()   # plain LM loss, standard Trainer
model.save_pretrained("out/"); PeftModel.from_pretrained(base_model, "out/").generate(...)
```

## 3. Config — flat, `LoraConfig`-idiomatic

The paper uses **uniform** settings across the three MLP projections, so `GaDRAConfig(PeftConfig)` is flat (not a
nested per-module map): LoRA-style `r`, `lora_alpha`, `lora_dropout`, `target_modules` (+ optional peft-standard
`rank_pattern`/`layers_to_transform` for future heterogeneity), plus GaDRA fields `router_conditioning ∈ {dual, mono}`,
`gate ∈ {hard, soft}`, `gamma_threshold`, `tau`, `gate_bias_init` (positive). Ported `from_dict` tolerates and **drops**
the legacy nested/regularizer/variant fields (used by the converter).

## 4. Registration — Decision A (locked)

`register_peft_method` requires the `PeftType` enum member to pre-exist (`peft/utils/peft_types.py:140-141`).
`gadra/_register.py` injects `GADRA` into `PeftType` at import (idempotent + peft-version guard) then calls
`register_peft_method(name="gadra", config_cls=GaDRAConfig, model_cls=GaDRAModel, prefix="gadra_")`. Write
`GaDRAConfig/GaDRALinear/GaDRAModel` in peft `tuners/lora/` style so they drop into `peft/tuners/gadra/` for a future
upstream PR. `GaDRAModel.prefix = "gadra_"`.

## 5. Layer forward (parity-critical)

```
base_out = base_layer(x)                  # frozen
if disabled: return base_out
x_d = dropout(x)                          # adapter-input dropout
delta = (lora_alpha/r) * gadra_B(gadra_A(x_d))   # A: kaiming, B: zeros; no bias; peft alpha/r scaling
sig   = delta if conditioning=="mono" else cat(base_out, delta)   # dual = [y0; delta]
z     = gadra_gate(dropout(sig))          # affine -> scalar; b_g init positive (gates start open)
gamma = hard:  train -> 1[gumbel_sigmoid(z,tau,thr) ]  (STE);  eval -> 1[sigmoid(z) > thr]
        soft:  <replicate code-as-run softplus path; confirm vs a soft run config in P2>
return base_out + gamma * delta           # gamma is per-token scalar (broadcast over d_out)
```
`gumbel_sigmoid` (the Gumbel-STE) is kept (method-critical). The regularizer/CR/R2/override/diagnostic code is dropped.

## 6. Remaining go-conditions

| # | Condition | Fix |
|---|---|---|
| R6 | old keys `...mlp.peft_blocks.<mod>.adapter.{0,1}` / `.gating.0` ≠ wrap-the-Linear layout | validated-bijection batch converter (§7) |
| soft-impl | paper soft `σ∈[0,1]` vs code `softplus` | replicate the code path that produced Tab.6; confirm vs a soft checkpoint's config in P2; document |
| b_g init | paper: gate bias positive | verify in code; replicate the published behavior (tell user if code≠paper) |
| Parity | eval bit-exact; training within tolerance | §9 |

(R1 reentrant, R2 key-parsing, R3 intervention order, R5 nested-config, R7 mlp-exclusivity — all **eliminated** by the v3 scope.)

## 7. Checkpoint back-compat — one-time BATCH conversion (locked)

Old `LoRA/{peft_config.json, peft_model.bin}` → new `{adapter_config.json, adapter_model.safetensors}`:
- Keys: `...mlp.peft_blocks.<mod>.adapter.{0,1}` → `...mlp.<mod>.gadra_A/gadra_B`; `gating.0` → `gadra_gate`; insert `.default`.
- Config: map nested→flat `GaDRAConfig`; drop regularizer/variant/svd fields. Detect `router_type` MA/UniPELT →
  `router_conditioning` dual/mono; `gamma_hard_masking` Gumbel→hard, None→soft.
- Originals read-only; new tree + manifest + checksums; assert key bijection + identical shapes; round-trip via
  `set_peft_model_state_dict`; old-vs-new eval logits parity on fixed fp32 inputs. No convert-on-load (legacy detector
  errors → converter). Convert `gadra-artifact/` once. (Drop any MoE/`mlp_external` checkpoints — out of scope.)

## 8. Phased plan (offline gates; GPU only at the end)

| Phase | Content | Offline gate |
|---|---|---|
| P0 | freeze golden parity baseline from current `PeftBlock` forward (dual/mono × hard/soft) | golden committed; ruff |
| P1 | `config.py` (flat) + `_register.py` (enum shim + register) | round-trip every GaDRA `peft_config.json` (drop dropped fields); py_compile; ruff |
| P2 | `gate.py` (`gumbel_sigmoid` + soft + hard) + `GaDRALinear` + `GaDRAModel` (inject up/gate/down, freeze, b_g+ init) | **forward parity 7a** (dual/mono × hard/soft) `max|Δ|<tol`; **logits parity 7b**; confirm soft-impl + b_g init |
| P3 | `compat/convert.py` batch converter | bijection + shape identity + round-trip |
| P4 | examples: `train_gadra.py` (standard `Trainer`, §A.1 recipe) + `generate.py`; README/usage docs | full CPU pytest green; ruff |
| P5 | execution-host end-to-end old-vs-new (HITL) on a representative GaDRA CPT config | canonical metrics within tolerance → retire old core |

## 9. Numeric-parity protocol (before == after)

- **Eval / inference: BIT-EXACT** — eval-mode Gumbel deterministic, no dropout; copy weights old→new; `atol=1e-6` fp32.
- **Training: within tolerance T** — per gated module: 1 Gumbel `rand_like` + 2 dropout draws → RNG order differs across
  stacks. Controlled-RNG unit parity `atol=1e-6` fp32; bf16 `atol=1e-3`; end-to-end CE curve rel `Δ<1e-3`, metrics `±0.1`.
- **Layers:** 7a CPU `PeftBlock` vs `GaDRALinear` (dual/mono × hard/soft × max_gating); 7b whole-model logits on a
  2-layer toy; 7c single-step CE loss + grads. (No aux-loss/loss_breakdown/stash/reentrant/ZeRO-3 threats.)

## 10. Package layout (v3)

```
GaDRA/
├── pyproject.toml          # peft==0.16.0, transformers==4.53.3 ; where=["src"] -> method-only wheel
├── README.md ; docs/DESIGN.md ; requirements-repro.txt
├── src/gadra/              # THE WHEEL = method only (`pip install gadra`)
│   ├── __init__.py         # imports _register (registers "gadra")
│   ├── _register.py        # PeftType enum shim + register_peft_method
│   ├── config.py           # GaDRAConfig(PeftConfig) — flat, LoraConfig-idiomatic
│   ├── gate.py             # gumbel_sigmoid + hard/soft gamma functions (method-critical)
│   ├── layer.py            # GaDRALinear(nn.Module, BaseTunerLayer) — gated residual forward
│   └── model.py            # GaDRAModel(BaseTuner): _create_and_replace, freeze, save/load
├── examples/               # repo-only reproduction tooling (NOT in the wheel; uv-managed, config-driven)
│   ├── train.py inference.py            # single config-driven entries (python -m examples.train / .inference)
│   ├── processing.py evaluation.py convert.py   # data packing · scorers+runner · legacy converter (verbatim)
│   ├── config/             # train.yaml · inference.yaml · deepspeed_zero2.yaml · llama jinja
│   ├── slurm/train.slurm   # uv multi-GPU SLURM template (accelerate + ZeRO-2)
│   └── tests/              # processing + evaluation + converter parity tests (+ their goldens)
└── tests/                  # method tests: layer golden/parity (7a/7b/7c) + independence + wheel-purity
```
Dropped vs v2: `mlp_wrapper.py`, `losses.py`, `collect.py`, `generation.py`, `intervention.py`, `trainer.py`.
The converter (`compat/`) and data/eval moved to the repo-only `examples/` tree (package/repo split, §0.3 of the plan).

## 11. Locked decisions

- Scope = paper-faithful method + training; MoE/mlp_external **removed**; variants = GaDRA + Mono + Soft (+ Mono-Soft). [locked]
- No aux loss → pure adapter, peft-default `use_reentrant`, standard `Trainer`, no sidecar. [resolved]
- Registration = A (`register_peft_method` + upstream-ready enum shim). [locked]
- Config = flat / `LoraConfig`-idiomatic. [locked]
- Back-compat = one-time batch conversion (method weights only). [locked]
- Merge = refuse uniformly. [locked]
- Open verification items (P2): soft-gate impl (softplus vs σ) and `b_g` positive init — replicate the code that produced the published numbers; document any paper-text mismatch.
