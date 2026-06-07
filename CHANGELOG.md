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

[Unreleased]: https://github.com/GlycerinLOL/GaDRA/commits/main
