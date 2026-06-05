"""P1 gate — GaDRAConfig defaults, serialization round-trip, legacy ingestion, scope guards.

These tests do not require the ``gadra`` method to be registered (P2), so they exercise the config
layer in isolation: enum shim, dataclass round-trip, and ``from_legacy_peft_config`` mapping/guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gadra import GaDRAConfig
from gadra._enum_shim import ensure_gadra_peft_type
from peft.utils.peft_types import PeftType

# Canonical legacy GaDRA configs in the source research repo.
_SOURCE_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _SOURCE_REPO / "configs"


def test_enum_shim_idempotent_and_registered_in_enum():
    a = ensure_gadra_peft_type()
    b = ensure_gadra_peft_type()
    assert a is b
    assert PeftType.GADRA is a
    assert "GADRA" in PeftType.__members__
    # register_peft_method's precondition: name.upper() must appear in list(PeftType).
    assert "GADRA" in list(PeftType)


def test_defaults_are_paper_faithful():
    cfg = GaDRAConfig()
    assert cfg.peft_type is PeftType.GADRA
    assert cfg.r == 512
    assert cfg.lora_alpha == 1.0
    assert cfg.target_modules == ["up_proj", "gate_proj", "down_proj"]
    assert cfg.router_conditioning == "dual"
    assert cfg.gate == "hard"
    assert cfg.gate_bias_init is None  # released-code behavior (default nn.Linear init)


@pytest.mark.parametrize("conditioning", ["dual", "mono"])
@pytest.mark.parametrize("gate", ["hard", "soft"])
def test_dict_roundtrip(conditioning, gate):
    cfg = GaDRAConfig(r=64, router_conditioning=conditioning, gate=gate)
    rebuilt = GaDRAConfig(**cfg.to_dict())
    assert rebuilt == cfg


def test_save_pretrained_writes_gadra_peft_type(tmp_path):
    cfg = GaDRAConfig(r=128)
    cfg.save_pretrained(str(tmp_path))
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["peft_type"] == "GADRA"
    assert saved["r"] == 128
    assert saved["target_modules"] == ["up_proj", "gate_proj", "down_proj"]


def test_invalid_choices_raise():
    with pytest.raises(ValueError):
        GaDRAConfig(router_conditioning="triple")
    with pytest.raises(ValueError):
        GaDRAConfig(gate="medium")
    with pytest.raises(ValueError):
        GaDRAConfig(r=0)


def test_from_legacy_dual_and_mono():
    dual = GaDRAConfig.from_legacy_peft_config(json.loads((_CONFIGS / "peft_config.json").read_text()))
    assert dual.router_conditioning == "dual"
    assert dual.gate == "hard"
    assert dual.r == 512
    assert dual.lora_alpha == 1.0
    assert dual.lora_dropout == 0.05
    assert set(dual.target_modules) == {"up_proj", "down_proj", "gate_proj"}

    mono = GaDRAConfig.from_legacy_peft_config(json.loads((_CONFIGS / "peft_config_mono.json").read_text()))
    assert mono.router_conditioning == "mono"
    assert mono.gate == "hard"


@pytest.mark.parametrize("fname", ["peft_config_ste.json", "peft_config_mlp_external.json", "peft_config_milora.json"])
def test_from_legacy_rejects_out_of_scope(fname):
    path = _CONFIGS / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    with pytest.raises(ValueError):
        GaDRAConfig.from_legacy_peft_config(json.loads(path.read_text()))


def _legacy_module(**overrides):
    spec = {
        "peft_rank": 512,
        "peft_type": "lora",
        "init_method": "default",
        "activation": "none",
        "router_type": "MA",
        "router_input": "default",
        "output_scalar": 1.0,
        "max_gating": False,
        "gamma_hard_masking": "Gumbel",
    }
    spec.update(overrides)
    return spec


def test_from_legacy_rejects_attention_target():
    legacy = {"gate_layers": None, "peft_layers": "all", "target_modules": {"q_proj": _legacy_module()}}
    with pytest.raises(ValueError):
        GaDRAConfig.from_legacy_peft_config(legacy)


def test_from_legacy_rejects_nonunit_output_scalar():
    legacy = {
        "gate_layers": None,
        "peft_layers": "all",
        "target_modules": {mod: _legacy_module(output_scalar=2.0) for mod in ("up_proj", "down_proj", "gate_proj")},
    }
    with pytest.raises(ValueError):
        GaDRAConfig.from_legacy_peft_config(legacy)
