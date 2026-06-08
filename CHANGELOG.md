# Changelog

All notable changes to GaDRA are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project will adopt
[Semantic Versioning](https://semver.org/) once it reaches `0.1.0`.

## [Unreleased]

### Added
- Initial public release of the GaDRA `peft`-native method (`src/gadra`): a custom `BaseTuner` /
  `BaseTunerLayer` registered via `register_peft_method`, used through the standard peft API
  (`get_peft_model`, `PeftModel.from_pretrained`, `save_pretrained`). Variants: GaDRA (dual+hard),
  GaDRA-Mono (mono+hard), GaDRA-Soft (dual+soft), GaDRA-Mono-Soft. Non-mergeable by design.
- Repo-only reproduction tooling under `examples/`: parity-exact data pipeline (`processing.py`),
  deterministic scorers + GPT judge (`evaluation.py`), legacy-checkpoint converter (`convert.py`), and
  config-driven `train.py` / `inference.py` with `examples/config/`.
- Unified, config-driven training entry: `examples/train.py` selects the adapter from the run-config's
  `method:` field (`gadra` | `gadra-mono` | `gadra-soft` | `lora`), dispatched by a small in-tree registry
  (`examples/methods.py`); the recipe is method-agnostic, so every method shares one trainer code path.
  Adding a baseline = one `@register` call. `examples/slurm/train.slurm` gains `METHOD` / `CONFIG` knobs.
  `examples/train_lora_baseline.py` is now a deprecated back-compat shim that forwards to
  `examples.train --override method=lora`. CPU-only construction-parity tests in `examples/tests/test_methods.py`.
- uv-managed reproduction environment (cu124 torch + prebuilt flash-attn + DeepSpeed ZeRO-2; `uv.lock`
  pinned) and SLURM wrappers (`examples/slurm/train.slurm` multi-GPU, `examples/slurm/inference.slurm`).
- Tests + CI: method-only purity / independence gates, layer / data / scorer parity goldens, and CI-safe
  inference-helper unit tests; GitHub Actions runs ruff + pytest + `uv lock --check`.
- Project metadata: Apache-2.0 `LICENSE`, `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/FAQ.md`.

### Changed
- **GaDRA now uses the standard peft `lora_alpha / r` scaling** (previously `lora_alpha` was applied
  directly to the delta with no `/r`). `lora_alpha` now means exactly what it does in `peft.LoraConfig`, so
  GaDRA and the native LoRA baseline share one convention. The default `lora_alpha` is `r` (via `None → r`
  in `__post_init__`), giving an effective residual scale of `1.0` — numerically identical to the previous
  `lora_alpha=1.0` direct default, so trained-model behavior is unchanged. The paper's residual scale α maps
  to `lora_alpha / r` (configs ship `lora_alpha = r`, e.g. 512). `from_legacy_peft_config` now emits
  `lora_alpha = output_scalar * r`; `examples/config/train.yaml` ships `lora_alpha: 512`. Saved
  `adapter_config.json` files written under the old direct convention (`lora_alpha: 1.0`) must be
  regenerated; the public release ships no checkpoints, so there is no external breakage.

### Fixed
- LoRA baseline scaling. The baseline config (`examples/config/train_lora.yaml`) carried `lora_alpha: 1.0`
  — but a stock `peft.LoraConfig` scales by `lora_alpha / r`, so `alpha=1.0, r=512` gave an effective scale
  of `1/512` (a ~512× weaker, near-no-op adapter). Now `lora_alpha = r` (512) → effective scale `1.0`,
  matching the paper's Fixed1.0 LoRA baseline (`output_scalar=1.0`). Only the standalone LoRA baseline was
  affected; the paper's reported LoRA numbers come from the research repo's (correct) `output_scalar=1.0`
  baseline.
- peft-conformance hardening (full audit of `src/gadra` + the examples LoRA baseline against peft 0.16.0):
  - `examples/methods.py` now forwards the standard optional `LoraConfig` knobs (`use_rslora`, `use_dora`,
    `init_lora_weights`, `bias`, `lora_bias`, `modules_to_save`) for the LoRA baseline instead of silently
    dropping them, and types `lora_alpha` as `int` to match peft.
  - `GaDRAConfig` gains the standard `init_lora_weights` and `exclude_modules` fields; the class docstring now
    lists the intentional deviations from `LoraConfig` (ordered `target_modules`, unsupported
    `layers_to_transform`/`layers_pattern`).
  - `GaDRALinear`: renamed the init flag `init_weights` → peft's `init_lora_weights` (honoring `False` =
    random `B`), uses a peft-standard `scaling` (`lora_alpha / r`) dict in `forward`, and added a `**kwargs`
    sink to `update_layer` that rejects unsupported peft LoRA options (`use_rslora`/`use_dora`/`lora_bias`)
    instead of silently ignoring them.
  - `GaDRAModel`: added attributable `NotImplementedError` stubs for `unload` / `delete_adapter` /
    `add_weighted_adapter`; made `_prepare_adapter_config` a `@staticmethod` with a documented rationale.
  - `_register.py`: dropped the redundant `_registered` flag — peft's registry is the single idempotency
    source of truth. `from_legacy_peft_config` now warns on unrecognized legacy top-level keys.
  - Audit confirmed (no change needed): `modules_to_save` already stays trainable, the `get_peft_model` →
    `save_pretrained` → `from_pretrained` round-trip is bit-exact, and registration/enum-shim are sound.

[Unreleased]: https://github.com/GlycerinLOL/GaDRA/commits/main
