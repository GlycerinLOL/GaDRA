# Changelog

All notable changes to GaDRA are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Dependency upgrade for newest-model support: `transformers` 4.53.3 → 5.10.2 and `peft` 0.16.0 →
  0.19.1 (torch floor 2.1 → 2.4). The `src/gadra` tuner needs no functional change — the peft
  custom-tuner registration path (`register_peft_method`, the `PeftType` enum shim, and the
  `BaseTuner` / `BaseTunerLayer` contract) is unchanged across these versions. transformers v5
  requires peft ≥ 0.18; MoE backbones require peft ≥ 0.19.

### Fixed
- `examples/train.py`: dropped `TrainingArguments(overwrite_output_dir=...)` (removed in
  transformers v5) and renamed the `from_pretrained` `torch_dtype=` argument to `dtype=`; same
  `torch_dtype` → `dtype` rename in `examples/inference.py`.

### Upgrading

Refresh your environment to pick up the new pins:

```bash
# uv (recommended) — method only, or add --group gpu for the reproduction stack
# (prebuilt flash-attn + DeepSpeed):
uv sync
uv sync --group gpu

# pip (portable reproduction path):
pip install -U -e .                                            # from a clone
pip install -U "git+https://github.com/GlycerinLOL/GaDRA.git"  # method only, no clone
```

Existing GaDRA adapters load unchanged under peft 0.19.1 — no retraining required.

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
