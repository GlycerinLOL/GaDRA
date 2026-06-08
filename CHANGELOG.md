# Changelog

All notable changes to GaDRA are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-08

Initial public release.

### Added
- The GaDRA `peft`-native method (`src/gadra`): a custom `BaseTuner` / `BaseTunerLayer` registered
  via `register_peft_method`, used through the standard peft API (`get_peft_model`,
  `PeftModel.from_pretrained`, `save_pretrained`). Variants: GaDRA (dual+hard), GaDRA-Mono
  (mono+hard), GaDRA-Soft (dual+soft), GaDRA-Mono-Soft. Standard peft `lora_alpha / r` scaling
  (default `lora_alpha = r`, unit residual scale). Model-agnostic via `target_modules`
  name-matching (validated on Llama-3.1-8B-Instruct and Qwen3-8B). Non-mergeable by design.
- Repo-only reproduction tooling under `examples/`: parity-exact data pipeline (`processing.py`),
  deterministic scorers + GPT judge (`evaluation.py`), legacy-checkpoint converter (`convert.py`),
  and config-driven `train.py` / `inference.py` with `examples/config/`. A single training entry
  selects the adapter from the run-config's `method:` field (`gadra` | `gadra-mono` | `gadra-soft`
  | `lora`); the recipe is method-agnostic, so every method shares one trainer code path.
- uv-managed reproduction environment (cu124 torch + prebuilt flash-attn + DeepSpeed ZeRO-2; pinned
  `uv.lock`) and SLURM wrappers (`examples/slurm/`).
- Tests + CI: method purity / independence gates, layer / data / scorer parity goldens, and
  CI-safe inference-helper unit tests; GitHub Actions runs ruff + pytest + `uv lock --check`.
- Documentation and project metadata: README, `docs/DESIGN.md`, `docs/FAQ.md`, Apache-2.0 `LICENSE`,
  `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`.

[1.0.0]: https://github.com/GlycerinLOL/GaDRA/releases/tag/v1.0.0
