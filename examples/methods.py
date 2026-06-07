"""Adapter-method registry for the unified training entry.

One ``method:`` field in the run-config selects the adapter family AND its variant defaults;
``build_peft_config(cfg)`` is the single seam that turns it into a constructed peft config object.
The training recipe (LR, schedule, batch size, packing, …) is method-agnostic and stays in the run
YAML — only the *adapter construction* lives here.

The examples ship four methods today:
  * ``gadra``       — GaDRA, the paper default (router_conditioning=dual, gate=hard)
  * ``gadra-mono``  — gate conditions on the delta only (router_conditioning=mono)
  * ``gadra-soft``  — continuous (soft) gate (gate=soft)
  * ``lora``        — stock ``peft.LoraConfig`` baseline (no gate)

A method is a named preset: a builder function + the knob defaults that define the variant. Per-knob
precedence is ``explicit cfg (YAML / --override)  >  variant preset  >  builder fallback``, so naming
``method: gadra-mono`` gives mono without touching the recipe, while an explicit ``router_conditioning``
in the YAML still wins. Adding a baseline = one ``@register`` call + a builder; no edits to
``train.py``, the SLURM wrapper, or ``load_run_config``.

Builders read NAMED cfg keys with defensive casts at the use site — the exact pattern (and the exact
constructor calls) the two entry scripts used before this unification — so construction is bit-faithful
to the previous code: ``method: gadra`` reproduces the old ``GaDRAConfig(...)`` call argument-for-
argument, and ``method: lora`` reproduces the old ``LoraConfig(...)`` call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

logger = logging.getLogger("gadra.methods")

# A builder takes (resolved run-config, the method's variant preset) and returns a constructed peft
# config object (GaDRAConfig | LoraConfig | any future PeftConfig). Heavy imports happen inside the
# builder body so ``import examples.methods`` stays cheap and ``method: lora`` never imports gadra.
PeftConfigBuilder = Callable[[Dict[str, Any], Dict[str, Any]], Any]

# GaDRA-only adapter knobs (top-level cfg keys). Setting any of these under method=lora is a config
# error (a silently-ignored adapter knob can quietly corrupt an experiment's meaning).
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
    """Decorator: register a builder under ``name`` (adding a baseline = one ``@register`` call)."""

    def _wrap(fn: PeftConfigBuilder) -> PeftConfigBuilder:
        if name in _REGISTRY:
            raise ValueError(f"method {name!r} already registered")
        _REGISTRY[name] = Method(name=name, builder=fn, defaults=dict(defaults or {}), summary=summary)
        return fn

    return _wrap


def available() -> list[str]:
    """Sorted list of registered method names (for error messages / introspection)."""
    return sorted(_REGISTRY)


def _knob(cfg: Dict[str, Any], preset: Dict[str, Any], key: str, fallback: Any) -> Any:
    """Per-knob precedence: explicit cfg (YAML / --override) > variant preset > builder fallback."""
    if cfg.get(key) is not None:
        return cfg[key]
    if preset.get(key) is not None:
        return preset[key]
    return fallback


def _shared(cfg: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    """The LoRA-family adapter hyperparams every builder consumes (defaults mirror the entry scripts)."""
    return dict(
        r=int(_knob(cfg, preset, "r", 512)),
        lora_alpha=float(_knob(cfg, preset, "lora_alpha", 1.0)),
        lora_dropout=float(_knob(cfg, preset, "lora_dropout", 0.05)),
        target_modules=list(_knob(cfg, preset, "target_modules", ["up_proj", "gate_proj", "down_proj"])),
    )


# --- builders: each reproduces the corresponding entry script's constructor call exactly ------------


@register("gadra", defaults={"router_conditioning": "dual", "gate": "hard"},
          summary="GaDRA (paper default): dual-conditioned gate, hard Gumbel-STE.")
@register("gadra-mono", defaults={"router_conditioning": "mono", "gate": "hard"},
          summary="GaDRA-Mono: gate conditions on the delta only.")
@register("gadra-soft", defaults={"router_conditioning": "dual", "gate": "soft"},
          summary="GaDRA-Soft: continuous (soft) gate.")
def _build_gadra(cfg: Dict[str, Any], preset: Dict[str, Any]) -> Any:
    from gadra import GaDRAConfig  # lazy: importing the package also registers the "gadra" peft method

    return GaDRAConfig(
        **_shared(cfg, preset),
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
    return LoraConfig(**_shared(cfg, preset), task_type="CAUSAL_LM")


def build_peft_config(cfg: Dict[str, Any]) -> Any:
    """Turn ``cfg['method']`` (default ``gadra``) into the right peft config object. THE dispatch.

    Default is ``gadra`` so a run-config with no ``method:`` reproduces the previous train.yaml behavior.
    """
    name = str(cfg.get("method", "gadra")).strip().lower()
    method = _REGISTRY.get(name)
    if method is None:
        raise ValueError(
            f"unknown method {name!r}; available: {', '.join(available())} "
            "(register a builder in examples/methods.py to add one)"
        )
    logger.info("method=%s", name)
    return method.builder(cfg, method.defaults)
