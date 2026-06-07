"""CPU-only construction-parity tests for the method registry (no GPU / data / model download).

These pin that the unified ``method:`` dispatch (examples/methods.py) builds the SAME peft config the
two former entry scripts built, and that the variant presets + the lora guard behave as designed.
"""

from __future__ import annotations

import pytest

from examples.methods import available, build_peft_config


def _base(**extra):
    """A minimal adapter-knob config (the shared LoRA-family fields); add a `method` etc. via kwargs."""
    cfg = dict(r=512, lora_alpha=1.0, lora_dropout=0.05,
               target_modules=["up_proj", "gate_proj", "down_proj"])
    cfg.update(extra)
    return cfg


def test_available_lists_shipped_methods():
    assert set(available()) >= {"gadra", "gadra-mono", "gadra-soft", "lora"}


def test_default_method_is_gadra_dual_hard():
    pc = build_peft_config(_base())  # no `method` key -> default gadra
    assert type(pc).__name__ == "GaDRAConfig"
    assert pc.router_conditioning == "dual"
    assert pc.gate == "hard"


@pytest.mark.parametrize("method,router,gate", [
    ("gadra", "dual", "hard"),
    ("gadra-mono", "mono", "hard"),
    ("gadra-soft", "dual", "soft"),
])
def test_gadra_presets(method, router, gate):
    pc = build_peft_config(_base(method=method))
    assert type(pc).__name__ == "GaDRAConfig"
    assert pc.router_conditioning == router
    assert pc.gate == gate


def test_shared_adapter_fields_are_faithful():
    pc = build_peft_config(_base(method="gadra"))
    assert pc.r == 512
    assert pc.lora_alpha == 1.0
    assert pc.lora_dropout == 0.05
    assert list(pc.target_modules) == ["up_proj", "gate_proj", "down_proj"]


def test_explicit_knob_overrides_preset():
    # An explicit cfg value (YAML / --override) wins over the variant preset.
    pc = build_peft_config(_base(method="gadra-mono", router_conditioning="dual"))
    assert pc.router_conditioning == "dual"


def test_method_is_case_insensitive():
    pc = build_peft_config(_base(method="GaDRA-Mono"))
    assert pc.router_conditioning == "mono"


def test_lora_builds_stock_loraconfig():
    pc = build_peft_config(_base(method="lora"))
    assert type(pc).__name__ == "LoraConfig"
    assert not hasattr(pc, "gate")
    assert not hasattr(pc, "router_conditioning")


@pytest.mark.parametrize("knob", ["router_conditioning", "gate", "gamma_threshold", "tau", "gate_bias_init"])
def test_lora_rejects_gadra_only_knobs(knob):
    with pytest.raises(ValueError, match="GaDRA-only"):
        build_peft_config(_base(method="lora", **{knob: "x"}))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        build_peft_config(_base(method="nope"))
