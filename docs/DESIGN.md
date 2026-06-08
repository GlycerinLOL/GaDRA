# GaDRA — Design & Architecture

How the GaDRA method maps to a HuggingFace `peft`-native custom tuner (`BaseTuner` /
`BaseTunerLayer`), and the numeric-parity protocol against the original research implementation.
Pinned environment: PyTorch 2.6.0, Transformers 5.7.0, peft 0.19.1, Python 3.12.

For *using* the method, see the [README → Quick start](../README.md#quick-start). This document is
the architecture reference.

## Scope

GaDRA is a `peft`-native custom tuner — model-agnostic (any HF `*ForCausalLM` via `target_modules`
name-matching), pip-installable, driven through the standard peft API (`get_peft_model`,
`save_pretrained`, `from_pretrained`) and the standard `transformers.Trainer` (plain LM loss, no
custom loop, no auxiliary loss).

| Element | Spec |
|---|---|
| Adapter | LoRA, `r=512`, `lora_alpha=r` (residual scale `α = lora_alpha/r = 1.0`), on `up_proj` / `gate_proj` / `down_proj` of every decoder block |
| Gate | one affine router per module `g = wᵀ[·] + b_g`, `b_g` init positive (gates start open) |
| Conditioning | `dual` `g([y⁰; δ])` (GaDRA) or `mono` `g(δ)` (GaDRA-Mono) |
| Gate type | `hard` binary via Gumbel-sigmoid STE (τ=1, no anneal), inference `1[σ(z)>0.5]`; `soft` continuous ablation |
| Output | `y = y⁰ + γ·δ`, `δ = (lora_alpha/r)·B·A·x`; non-mergeable (γ per-token, input-dependent) |
| Training | standard `Trainer`, plain LM loss; AdamW, cosine LR, BF16, DeepSpeed ZeRO-2, FA2, seqlen 1024, gbs 128 / mb 16, clip 0.1, wd 0.01, seed 42, 3 epochs |
| Backbones | Llama-3.1-8B-Instruct (primary), Qwen3-8B (probe); name-matching covers any HF model |
| Variants | GaDRA (dual+hard), GaDRA-Mono (mono+hard), GaDRA-Soft (dual+soft), GaDRA-Mono-Soft |

The tuner is generic: name-matching can wrap any `nn.Linear` (attention included), but the
paper-faithful default targets the three MLP projections.

The reproduction tooling (data pipeline, scorers, converter, entry scripts) lives in the repo-only
`examples/` tree and is **not** shipped by `pip install gadra`; the package is the method only and
never imports `examples/`. See [examples/README.md](../examples/README.md).

## Layer forward (parity-critical)

```
base_out = base_layer(x)                                  # frozen
if disabled: return base_out
x_d   = dropout(x)                                        # adapter-input dropout
delta = (lora_alpha / r) * gadra_B(gadra_A(x_d))          # A: kaiming, B: zeros; no bias
sig   = delta if conditioning == "mono" else cat(base_out, delta)   # dual = [y⁰; δ]
z     = gadra_gate(dropout(sig))                          # affine -> per-token scalar; b_g init positive
gamma = hard: train -> gumbel_sigmoid(z, τ, thr) (STE);  eval -> 1[σ(z) > thr]
        soft: softplus(z)
return base_out + gamma * delta                           # γ is a per-token scalar (broadcast over d_out)
```

`gumbel_sigmoid` is the method-critical Gumbel straight-through estimator. When the gate closes
(`γ=0`) the output is bit-identical to the frozen base.

> **Soft gate:** the released code path that produced the paper's soft-gate numbers is `softplus(z)`,
> while the paper text describes a `σ ∈ [0,1]` gate. The code-as-run `softplus` is reproduced here.

## Config

`GaDRAConfig(PeftConfig)` is flat (the paper uses uniform settings across the three projections):
LoRA-style `r` / `lora_alpha` / `lora_dropout` / `target_modules` (+ optional peft-standard
`rank_pattern` / `alpha_pattern` / `exclude_modules`), plus the GaDRA fields
`router_conditioning ∈ {dual, mono}`, `gate ∈ {hard, soft}`, `gamma_threshold`, `tau`,
`gate_bias_init`. `from_legacy_peft_config` ingests a legacy nested config and drops the
regularizer / analysis / out-of-scope-variant fields (used by the converter).

## Registration

`register_peft_method` requires the `PeftType` enum member to pre-exist. `gadra/_enum_shim.py`
injects `GADRA` into `PeftType` at import (idempotent, peft-version-guarded); `gadra/_register.py`
then calls `register_peft_method(name="gadra", config_cls=GaDRAConfig, model_cls=GaDRAModel,
prefix="gadra_")`. `GaDRAConfig` / `GaDRALinear` / `GaDRAModel` follow peft's `tuners/lora/` style,
so they could drop into `peft/tuners/gadra/` for a future upstream PR.

## Checkpoint conversion

`examples/convert.py` converts a legacy research-code checkpoint
(`LoRA/{peft_config.json, peft_model.bin}`) to the peft-native format
(`{adapter_config.json, adapter_model.safetensors}`):

- Keys: `…mlp.peft_blocks.<mod>.adapter.{0,1}` → `…mlp.<mod>.gadra_{A,B}`; `gating.0` → `gadra_gate`
  (peft re-inserts `.default` on load).
- Config: legacy nested → flat `GaDRAConfig`; `router_type` MA/UniPELT → `router_conditioning`
  dual/mono; `gamma_hard_masking` Gumbel/None → `gate` hard/soft.
- Originals are read-only; the converter asserts a key bijection + identical shapes and writes a
  manifest. Out-of-scope checkpoints (whole-MLP, attention, `svd_minor`, STE) are rejected.

## Numeric-parity protocol

- **Eval / inference — bit-exact.** Eval-mode Gumbel is deterministic and dropout is off; copying
  weights old→new gives `atol=1e-6` in fp32.
- **Training — within tolerance.** Each gated module draws 1 Gumbel `rand_like` + 2 dropout samples,
  so the RNG order differs across stacks; controlled-RNG unit parity holds at `atol=1e-6` (fp32) /
  `1e-3` (bf16), end-to-end CE within `Δ<1e-3`, metrics within `±0.1`.
- **Goldens:** `PeftBlock` vs `GaDRALinear` across `dual/mono × hard/soft`, whole-model logits on a
  toy model, and single-step CE loss + grads. Frozen baselines live in `tests/_golden/` and
  `examples/tests/_golden/` (`tests/test_parity.py`, `examples/tests/test_processing.py`).

## Package layout

```
src/gadra/                  # the pip-installable method (zero data/eval deps)
  config.py                 #   GaDRAConfig (flat, LoraConfig-idiomatic)
  layer.py / gate.py        #   GaDRALinear + the gate (Gumbel-STE / softplus)
  model.py                  #   GaDRAModel(BaseTuner): inject / freeze / save / load
  _register.py _enum_shim.py   # PeftType shim + register_peft_method wiring
examples/                   # repo-only reproduction tooling (NOT in the wheel)
  train.py inference.py     #   config-driven entries (train picks the adapter from method: via an in-file registry)
  processing.py evaluation.py convert.py   # data · scorers+judge · legacy converter
  config/ slurm/ tests/
tests/                      # method purity / parity / independence goldens
```

Out of scope (stays in the research repo): MoE / whole-MLP wrappers, attention-only variants,
alternate router/init variants (`Add`, `svd_minor`, relu/tanh), the manual `gate_layers` /
`peft_layers` DSL (the per-layer activation budget is *learned*), and the per-token analysis tooling
(CR/γ extraction, gate-override interventions, regularizer losses, score notebooks).
