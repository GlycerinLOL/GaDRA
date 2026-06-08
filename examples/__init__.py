"""GaDRA reproduction tooling — repo-only scripts (data processing, evaluation, checkpoint conversion,
and the config-driven train/inference entries). Not part of the pip-installable ``gadra`` package."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def load_run_config(config_path: str, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML run-config and apply ``key=value`` CLI overrides. Each override value is parsed as YAML."""
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
