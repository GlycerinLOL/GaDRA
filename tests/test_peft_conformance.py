"""CPU-only peft-conformance regression tests for the GaDRA tuner (no GPU / model download).

These pin the peft-API behaviors hardened in the conformance pass: ``init_lora_weights`` semantics,
``modules_to_save`` staying trainable, the introspection ``scaling`` attribute, and attributable
``NotImplementedError`` stubs for unsupported standard peft model ops.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

import gadra  # noqa: F401  (registers the "gadra" peft method)
from gadra import GaDRAConfig
from gadra.layer import GaDRALinear
from peft import get_peft_model


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.lm_head(self.up_proj(x))


def _peft(**cfg_kw):
    cfg = GaDRAConfig(r=4, target_modules=["up_proj"], **cfg_kw)
    return get_peft_model(_Toy(), cfg)


def _gadra_layer(pm):
    return next(m for m in pm.modules() if isinstance(m, GaDRALinear))


def test_modules_to_save_stays_trainable():
    # Regression: the modules_to_save copy must remain trainable after _mark_only_adapters_as_trainable.
    pm = _peft(modules_to_save=["lm_head"])
    ms = {n: p.requires_grad for n, p in pm.named_parameters() if "lm_head.modules_to_save" in n}
    assert ms and all(ms.values()), ms


def test_init_lora_weights_true_zeros_B():
    layer = _gadra_layer(_peft(init_lora_weights=True))
    assert layer.gadra_B["default"].weight.abs().sum().item() == 0.0


def test_init_lora_weights_false_gives_nonzero_B():
    layer = _gadra_layer(_peft(init_lora_weights=False))
    assert layer.gadra_B["default"].weight.abs().sum().item() > 0.0


def test_scaling_is_alpha_over_r():
    # peft-standard scaling = lora_alpha / r (used in forward, like LoraLayer).
    assert "scaling" in GaDRALinear.other_param_names
    assert _gadra_layer(_peft()).scaling["default"] == 1.0          # default lora_alpha = r (=4)
    assert _gadra_layer(_peft(lora_alpha=8)).scaling["default"] == 2.0  # 8 / r(4) = 2.0


def test_unsupported_lora_option_rejected_in_update_layer():
    layer = _gadra_layer(_peft())
    with pytest.raises(ValueError, match="use_rslora"):
        layer.update_layer("x", r=4, lora_alpha=1.0, lora_dropout=0.0,
                           router_conditioning="dual", gate="hard", use_rslora=True)


@pytest.mark.parametrize("op", ["unload", "delete_adapter", "add_weighted_adapter",
                                "merge_and_unload", "merge_adapter"])
def test_unsupported_model_ops_raise_notimplemented(op):
    tuner = _peft().base_model  # the GaDRAModel
    with pytest.raises(NotImplementedError):
        getattr(tuner, op)()


def test_installed_peft_matches_validated_version():
    # Env-drift guard: the running peft must match the version gadra's enum-shim / register and the
    # BaseTuner overrides were validated against, so a stale local env fails here, not only on CI.
    import peft

    from gadra._register import _EXPECTED_PEFT_VERSION

    assert peft.__version__ == _EXPECTED_PEFT_VERSION, (
        f"installed peft=={peft.__version__} != validated peft=={_EXPECTED_PEFT_VERSION}; sync the env "
        "to the pin (`uv sync` / `pip install -e .`), or if intentionally bumping update "
        "_EXPECTED_PEFT_VERSION in src/gadra/_register.py and the pin in pyproject.toml together."
    )
