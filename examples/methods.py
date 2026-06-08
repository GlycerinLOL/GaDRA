"""Adapter-method registry for the unified training entry.

The run-config's ``method:`` field selects the adapter family and its variant defaults;
``build_peft_config(cfg)`` turns it into a constructed peft config object. Four methods:
  * ``gadra``       — GaDRA (router_conditioning=dual, gate=hard)
  * ``gadra-mono``  — gate conditions on the delta only (router_conditioning=mono)
  * ``gadra-soft``  — continuous (soft) gate (gate=soft)
  * ``lora``        — stock ``peft.LoraConfig`` baseline (no gate)

Per-knob precedence: explicit cfg (YAML / --override) > variant preset > builder fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

logger = logging.getLogger("gadra.methods")

# A builder takes (resolved run-config, the method's variant preset) and returns a constructed peft
# config object (GaDRAConfig | LoraConfig).
PeftConfigBuilder = Callable[[Dict[str, Any], Dict[str, Any]], Any]

# GaDRA-only adapter knobs (top-level cfg keys); setting any under method=lora is a config error.
_GADRA_ONLY_KNOBS = ("router_conditioning", "gate", "gamma_threshold", "tau", "gate_bias_init")


@dataclass(frozen=True)
class Method:
    """A named adapter preset: a builder + the variant-defining knob defaults."""

    name: str
    builder: PeftConfigBuilder
    defaults: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


_REGISTRY: Dict[str, Method] = {}


def register(name: str, *, defaults: Dict[str, Any] | None = None, summary: str = ""):
    """Decorator: register a builder under ``name``."""

    def _wrap(fn: PeftConfigBuilder) -> PeftConfigBuilder:
        if name in _REGISTRY:
            raise ValueError(f"method {name!r} already registered")
        _REGISTRY[name] = Method(name=name, builder=fn, defaults=dict(defaults or {}), summary=summary)
        return fn

    return _wrap


def available() -> list[str]:
    """Sorted list of registered method names."""
    return sorted(_REGISTRY)


def _knob(cfg: Dict[str, Any], preset: Dict[str, Any], key: str, fallback: Any) -> Any:
    """Per-knob precedence: explicit cfg (YAML / --override) > variant preset > builder fallback."""
    if cfg.get(key) is not None:
        return cfg[key]
    if preset.get(key) is not None:
        return preset[key]
    return fallback


def _shared(cfg: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    """LoRA-family fields shared across methods (r / dropout / target_modules)."""
    return dict(
        r=int(_knob(cfg, preset, "r", 512)),
        lora_dropout=float(_knob(cfg, preset, "lora_dropout", 0.05)),
        target_modules=list(_knob(cfg, preset, "target_modules", ["up_proj", "gate_proj", "down_proj"])),
    )


@register("gadra", defaults={"router_conditioning": "dual", "gate": "hard"},
          summary="GaDRA: dual-conditioned gate, hard Gumbel-STE.")
@register("gadra-mono", defaults={"router_conditioning": "mono", "gate": "hard"},
          summary="GaDRA-Mono: gate conditions on the delta only.")
@register("gadra-soft", defaults={"router_conditioning": "dual", "gate": "soft"},
          summary="GaDRA-Soft: continuous (soft) gate.")
def _build_gadra(cfg: Dict[str, Any], preset: Dict[str, Any]) -> Any:
    from gadra import GaDRAConfig  # importing the package also registers the "gadra" peft method

    shared = _shared(cfg, preset)
    return GaDRAConfig(
        **shared,
        # default lora_alpha = r => effective scale 1.0.
        lora_alpha=float(_knob(cfg, preset, "lora_alpha", shared["r"])),
        router_conditioning=_knob(cfg, preset, "router_conditioning", "dual"),
        gate=_knob(cfg, preset, "gate", "hard"),
        task_type="CAUSAL_LM",
    )


@register("lora", summary="Stock peft.LoraConfig baseline (no gate).")
def _build_lora(cfg: Dict[str, Any], preset: Dict[str, Any]) -> Any:
    from peft import LoraConfig

    stray = [k for k in _GADRA_ONLY_KNOBS if cfg.get(k) is not None]
    if stray:
        raise ValueError(
            f"method='lora' is a stock LoRA baseline and takes no GaDRA-only knob(s) {stray}; "
            "remove them (router_conditioning / gate / gamma_threshold / tau / gate_bias_init are "
            "GaDRA-only), or choose a gadra* method."
        )
    shared = _shared(cfg, preset)
    # default alpha = r so the effective scale is 1.0 (native alpha/r scaling).
    lora_alpha = int(_knob(cfg, preset, "lora_alpha", shared["r"]))
    return LoraConfig(
        **shared,
        lora_alpha=lora_alpha,
        use_rslora=bool(_knob(cfg, preset, "use_rslora", False)),
        use_dora=bool(_knob(cfg, preset, "use_dora", False)),
        init_lora_weights=_knob(cfg, preset, "init_lora_weights", True),
        bias=_knob(cfg, preset, "bias", "none"),
        lora_bias=bool(_knob(cfg, preset, "lora_bias", False)),
        modules_to_save=_knob(cfg, preset, "modules_to_save", None),
        task_type="CAUSAL_LM",
    )


def build_peft_config(cfg: Dict[str, Any]) -> Any:
    """Turn ``cfg['method']`` (default ``gadra``) into the right peft config object."""
    name = str(cfg.get("method", "gadra")).strip().lower()
    method = _REGISTRY.get(name)
    if method is None:
        raise ValueError(
            f"unknown method {name!r}; available: {', '.join(available())} "
            "(register a builder in examples/methods.py to add one)"
        )
    logger.info("method=%s", name)
    return method.builder(cfg, method.defaults)
