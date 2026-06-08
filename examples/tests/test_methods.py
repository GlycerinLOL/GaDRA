"""Construction-parity tests for the method registry (``examples/train.py``)."""

from __future__ import annotations

import pytest

from examples.train import available, build_peft_config


def _base(**extra):
    """A minimal adapter-knob config; lora_alpha omitted so each method exercises its own default."""
    cfg = dict(r=512, lora_dropout=0.05, target_modules=["up_proj", "gate_proj", "down_proj"])
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
    assert pc.lora_alpha == 512  # alpha/r = 1.0
    assert pc.lora_dropout == 0.05
    assert list(pc.target_modules) == ["up_proj", "gate_proj", "down_proj"]


def test_explicit_knob_overrides_preset():
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


def test_gadra_alpha_defaults_to_r():
    # Default lora_alpha = r => effective scale 1.0.
    pc = build_peft_config(_base(method="gadra"))  # no explicit lora_alpha, r=512
    assert pc.lora_alpha == 512


def test_lora_alpha_defaults_to_r_for_unit_effective_scale():
    # Default alpha = r => effective scale 1.0; guards against accidental near-no-op alpha=1.0.
    pc = build_peft_config(_base(method="lora"))
    assert pc.lora_alpha == 512  # alpha/r = 1.0
    pc2 = build_peft_config(_base(method="lora", r=128))
    assert pc2.lora_alpha == 128  # alpha/r = 1.0


def test_lora_explicit_alpha_is_native_semantics():
    pc = build_peft_config(_base(method="lora", r=512, lora_alpha=1024))
    assert pc.lora_alpha == 1024


def test_lora_alpha_is_int_typed():
    assert isinstance(build_peft_config(_base(method="lora")).lora_alpha, int)


def test_lora_forwards_native_peft_knobs():
    pc = build_peft_config(_base(method="lora", use_rslora=True, init_lora_weights=False, bias="all"))
    assert pc.use_rslora is True
    assert pc.init_lora_weights is False
    assert pc.bias == "all"
    assert build_peft_config(_base(method="lora", use_dora=True)).use_dora is True
    assert build_peft_config(_base(method="lora", lora_bias=True)).lora_bias is True


@pytest.mark.parametrize("knob", ["router_conditioning", "gate", "gamma_threshold", "tau", "gate_bias_init"])
def test_lora_rejects_gadra_only_knobs(knob):
    with pytest.raises(ValueError, match="GaDRA-only"):
        build_peft_config(_base(method="lora", **{knob: "x"}))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        build_peft_config(_base(method="nope"))
