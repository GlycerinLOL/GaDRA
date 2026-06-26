"""Tests for the block-parallel target type (``GaDRAParallelBlock`` + ``parallel_modules`` dispatch).

Blocks are local ``nn.Module`` mocks that mirror the transformers 5.7.0 MoE-block contracts
(bare-tensor return, tuple return, extra positional arg) so the tests are version-agnostic.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import gadra  # noqa: F401  (registers the "gadra" peft method)
from gadra import GaDRAConfig
from gadra.layer import GaDRALinear, GaDRAParallelBlock
from peft import get_peft_model

H = 16
E = 8


class _DenseBlock(nn.Module):
    """LlamaMLP-like: tensor -> tensor."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, H)

    def forward(self, x):
        return self.lin(x)


class _TupleMoEBlock(nn.Module):
    """OlmoeSparseMoeBlock-like (4.53 contract): tensor -> (tensor, router_logits)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, E)
        self.experts = nn.Linear(H, H)

    def forward(self, x):
        return self.experts(x), self.gate(x)


class _ExtraArgBlock(nn.Module):
    """NLLB-MoE-like: (tensor, extra) -> tensor."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, H)

    def forward(self, x, extra):
        return self.lin(x) + 0.0 * extra.sum()


def _layer(block, *, gate="hard", router_conditioning="dual", init_lora_weights=True, **kw):
    return GaDRAParallelBlock(
        block, "default", hidden_size=H, r=4, lora_alpha=4, lora_dropout=0.0,
        router_conditioning=router_conditioning, gate=gate, init_lora_weights=init_lora_weights, **kw
    ).eval()


# --- layer: forward semantics ---

def test_dense_block_identity_at_init():
    block = _DenseBlock()
    layer = _layer(block)  # B zero-init => delta 0
    x = torch.randn(2, 3, H)
    assert torch.allclose(layer(x), block(x), atol=1e-6)


@pytest.mark.parametrize("gate", ["soft", "hard"])
def test_gate_active_changes_output(gate):
    block = _DenseBlock()
    layer = _layer(block, gate=gate, init_lora_weights=False, gate_bias_init=5.0)  # nonzero B, gate forced open
    x = torch.randn(2, 3, H)
    assert not torch.allclose(layer(x), block(x), atol=1e-5)


def test_tuple_block_preserves_router_logits():
    block = _TupleMoEBlock()
    layer = _layer(block, init_lora_weights=False)
    x = torch.randn(2, 3, H)
    raw = block(x)
    out = layer(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.allclose(out[1], raw[1], atol=1e-6)          # router_logits untouched
    assert not torch.allclose(out[0], raw[0], atol=1e-5)      # hidden adapted


def test_extra_positional_arg_passthrough():
    layer = _layer(_ExtraArgBlock(), router_conditioning="mono", gate="none")
    out = layer(torch.randn(2, 3, H), torch.ones(1))
    assert tuple(out.shape) == (2, 3, H)


def test_gate_none_is_ungated_low_rank_residual():
    block = _DenseBlock()
    layer = _layer(block, gate="none", init_lora_weights=False)
    assert "default" not in layer.gadra_gate                  # no gate sub-module built
    x = torch.randn(2, 3, H)
    scaling = layer.scaling["default"]
    delta = scaling * layer.gadra_B["default"](layer.gadra_A["default"](x))
    assert torch.allclose(layer(x), block(x) + delta, atol=1e-6)  # gamma == 1


@pytest.mark.parametrize("conditioning,expected_in", [("dual", 2 * H), ("mono", H)])
def test_gate_input_dim_matches_conditioning(conditioning, expected_in):
    layer = _layer(_DenseBlock(), router_conditioning=conditioning)
    assert layer.gadra_gate["default"].in_features == expected_in


def test_merge_and_unmerge_raise():
    layer = _layer(_DenseBlock())
    with pytest.raises(NotImplementedError):
        layer.merge()
    with pytest.raises(NotImplementedError):
        layer.unmerge()


def test_invalid_gate_and_conditioning_raise():
    with pytest.raises(ValueError):
        _layer(_DenseBlock(), gate="medium")
    with pytest.raises(ValueError):
        _layer(_DenseBlock(), router_conditioning="triple")


# --- model: dispatch / injection ---

class _Layer(nn.Module):
    def __init__(self, block_cls):
        super().__init__()
        self.mlp = block_cls()

    def forward(self, x):
        out = self.mlp(x)
        return out[0] if isinstance(out, tuple) else out


class _Model(nn.Module):
    def __init__(self, block_cls=_TupleMoEBlock, n=2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(block_cls) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_parallel_modules_targets_block_and_only_adapters_trainable():
    model = get_peft_model(_Model(), GaDRAConfig(parallel_modules=["mlp"], r=4))
    wrapped = [mod for name, mod in model.named_modules() if name.endswith(".mlp")]
    assert wrapped and all(isinstance(m, GaDRAParallelBlock) for m in wrapped)
    nongadra_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "gadra_" not in n]
    assert nongadra_trainable == []


def test_no_double_wrap_of_inner_modules():
    model = get_peft_model(_Model(), GaDRAConfig(parallel_modules=["mlp"], r=4))
    inner = [
        name for name, mod in model.named_modules()
        if isinstance(mod, (GaDRALinear, GaDRAParallelBlock)) and not name.endswith("mlp")
    ]
    assert inner == []


def test_auto_detect_wraps_nonlinear_target():
    # target_modules names a composite module and parallel_modules is unset -> auto block-parallel
    model = get_peft_model(_Model(), GaDRAConfig(target_modules=["mlp"], r=4))
    wrapped = [mod for name, mod in model.named_modules() if name.endswith(".mlp")]
    assert wrapped and all(isinstance(m, GaDRAParallelBlock) for m in wrapped)


def test_guard_rejects_unlisted_nonlinear_target():
    with pytest.raises(ValueError, match="parallel_modules"):
        get_peft_model(_Model(), GaDRAConfig(target_modules=["mlp"], parallel_modules=["other"], r=4))


class _FusedBlock(nn.Module):
    """v5 fused-expert-like: no nn.Linear, so hidden_size must come from the model config."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(H, H))

    def forward(self, x):
        return x @ self.w


def test_hidden_size_from_model_config_when_no_linear_to_probe():
    class _Cfg:
        hidden_size = H

        def to_dict(self):
            return {"hidden_size": H, "tie_word_embeddings": False}

    class _ConfModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.layers = nn.ModuleList([_Layer(_FusedBlock)])

        def forward(self, x):
            return self.layers[0](x)

    model = get_peft_model(_ConfModel(), GaDRAConfig(parallel_modules=["mlp"], r=4))
    layer = next(m for _, m in model.named_modules() if isinstance(m, GaDRAParallelBlock))
    assert layer.hidden_size == H


def test_linear_target_still_dispatches_to_gadralinear():
    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.up_proj(x)

    model = get_peft_model(_Toy(), GaDRAConfig(r=4, target_modules=["up_proj"]))
    assert isinstance(dict(model.named_modules())["base_model.model.up_proj"], GaDRALinear)


# --- config ---

def test_parallel_modules_config_defaults():
    assert GaDRAConfig(parallel_modules=["mlp"]).target_modules == []      # no per-projection default
    assert GaDRAConfig().target_modules == ["up_proj", "gate_proj", "down_proj"]
    GaDRAConfig(gate="none")  # accepted
    with pytest.raises(ValueError):
        GaDRAConfig(gate="medium")


def test_set_adapter_reaches_parallel_block():
    model = get_peft_model(_Model(), GaDRAConfig(parallel_modules=["mlp"], r=4))
    block = next(m for _, m in model.named_modules() if isinstance(m, GaDRAParallelBlock))
    block._active_adapter = ["stale"]                        # seed a stale active adapter
    model.base_model.set_adapter("default")                  # GaDRAModel.set_adapter must re-touch the block
    assert block.active_adapters == ["default"]


def test_gate_none_on_linear_target_is_ungated():
    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.up_proj(x)

    model = get_peft_model(_Toy(), GaDRAConfig(r=4, target_modules=["up_proj"], gate="none"))
    layer = dict(model.named_modules())["base_model.model.up_proj"]
    assert isinstance(layer, GaDRALinear)
    assert "default" not in layer.gadra_gate                 # ungated: no gate sub-module built
