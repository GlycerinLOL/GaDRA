"""GaDRA reproduction tooling — repo-only, NOT part of the pip-installable ``gadra`` package.

``pip install gadra`` ships the method alone (``src/gadra``). This ``examples`` tree (data processing,
paper-repro evaluation, the legacy-checkpoint converter, and the two config-driven entry scripts) lives
outside ``src/`` and is never built into the wheel. Use it from a repo checkout: ``uv sync --group gpu``
then ``python -m examples.train`` / ``python -m examples.inference``.

Layout (flat, config-driven):
  * ``examples/train.py``      — single training entry (reads ``config/train.yaml``)
  * ``examples/inference.py``  — single inference entry (reads ``config/inference.yaml``; modes generate|eval)
  * ``examples/processing.py`` — data: tokenizer + EOS + FA2 packing
  * ``examples/evaluation.py`` — eval: parsers + deterministic scorers + greedy runner + PPL
  * ``examples/convert.py``    — legacy→peft-native checkpoint converter
  * ``examples/config/``       — YAML run-configs + DeepSpeed ZeRO-2 + chat template

The dependency direction is one-way: ``examples`` imports the installed ``gadra`` method; the ``gadra``
package never imports ``examples``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def load_run_config(config_path: str, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML run-config and apply ``key=value`` CLI overrides (both entry scripts share this).

    Each override value is parsed as YAML, so ``--override per_device_train_batch_size=8`` yields an int,
    ``--override packing=group`` a str, ``--override target_modules=[up_proj,down_proj]`` a list. Argv-bound
    numerics that YAML 1.1 leaves as strings (e.g. ``1e-4``) are cast defensively at the use site, so both
    ``learning_rate: 1.0e-4`` (YAML float) and ``--override learning_rate=1e-4`` (str) work.
    """
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"run-config {config_path!r} must be a YAML mapping, got {type(cfg).__name__}")
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--override expects key=value, got {ov!r}")
        key, _, raw = ov.partition("=")
        cfg[key.strip()] = yaml.safe_load(raw.strip())
    return cfg
