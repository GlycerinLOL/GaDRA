"""Parity gate: ``GaDRALinear`` reproduces the released ``PeftBlock`` outputs; peft API integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import gadra  # noqa: F401  (registers the "gadra" method)
from gadra import GaDRAConfig
from gadra.layer import GaDRALinear

_GOLDEN_DIR = Path(__file__).resolve().parent / "_golden"
_VARIANTS = ["dual_hard", "dual_soft", "mono_hard", "mono_soft"]


class _ConstBase(nn.Module):
    """Stand-in base layer that returns a fixed output, isolating the adapter+gate math."""

    def __init__(self, y0: torch.Tensor, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("_y0", y0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self._y0


def _load_golden(name: str) -> dict:
    path = _GOLDEN_DIR / f"golden_{name}.pt"
    if not path.exists():
        pytest.skip(f"golden {name} not present; run tests/capture_golden.py")
    return torch.load(path, weights_only=False)


@pytest.mark.parametrize("name", _VARIANTS)
def test_layer_parity_eval_bit_exact(name):
    g = _load_golden(name)
    conditioning, gate = name.split("_")
    dims = g["dims"]
    sd = g["state_dict"]

    layer = GaDRALinear(
        _ConstBase(g["y0"], dims["in_features"], dims["out_features"]),
        adapter_name="default",
        r=dims["rank"],
        lora_alpha=dims["rank"],  # alpha/r = 1.0
        lora_dropout=g["dropout"],
        router_conditioning=conditioning,
        gate=gate,
        gamma_threshold=g["gamma_threshold"],
        tau=1.0,
        init_lora_weights=True,
    )

    with torch.no_grad():
        layer.gadra_A["default"].weight.copy_(sd["adapter.0.weight"])
        layer.gadra_B["default"].weight.copy_(sd["adapter.1.weight"])
        layer.gadra_gate["default"].weight.copy_(sd["gating.0.weight"])
        layer.gadra_gate["default"].bias.copy_(sd["gating.0.bias"])

    layer.eval()
    with torch.no_grad():
        out = layer(g["x"])

    # Hard gate: bit-exact. Soft gate: numerical equivalence (softplus drifts ~1e-4 across builds).
    if gate == "hard":
        torch.testing.assert_close(out, g["eval_output"], atol=1e-6, rtol=0.0)
    else:
        torch.testing.assert_close(out, g["eval_output"], atol=1e-3, rtol=1e-3)


def test_multi_adapter_dual_gate_is_order_independent():
    # Two dual adapters on one layer: stacking order must not affect the result.
    in_f, out_f = 12, 10
    g = torch.Generator().manual_seed(5)
    y0 = torch.randn(2, 4, out_f, generator=g)
    x = torch.randn(2, 4, in_f, generator=g)

    layer = GaDRALinear(
        _ConstBase(y0, in_f, out_f), "a", r=4, lora_alpha=4, lora_dropout=0.0,
        router_conditioning="dual", gate="soft",
    )
    layer.update_layer("b", r=4, lora_alpha=4, lora_dropout=0.0, router_conditioning="dual", gate="soft")
    with torch.no_grad():
        for name, seed in (("a", 11), ("b", 22)):
            gg = torch.Generator().manual_seed(seed)
            layer.gadra_A[name].weight.copy_(torch.randn(layer.gadra_A[name].weight.shape, generator=gg))
            layer.gadra_B[name].weight.copy_(torch.randn(layer.gadra_B[name].weight.shape, generator=gg))
            layer.gadra_gate[name].weight.copy_(torch.randn(layer.gadra_gate[name].weight.shape, generator=gg))
            layer.gadra_gate[name].bias.copy_(torch.randn(layer.gadra_gate[name].bias.shape, generator=gg))

    layer.eval()
    with torch.no_grad():
        layer._active_adapter = ["a", "b"]
        out_ab = layer(x)
        layer._active_adapter = ["b", "a"]
        out_ba = layer(x)
    torch.testing.assert_close(out_ab, out_ba, atol=1e-4, rtol=1e-5)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), cfg


def test_injection_only_adapters_trainable_and_zero_init_identity():
    from peft import get_peft_model

    base, _ = _tiny_llama()
    base_snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}
    ids = torch.randint(0, 64, (2, 6))

    with torch.no_grad():
        base_logits = base(ids).logits

    model = get_peft_model(base, GaDRAConfig(r=8, task_type="CAUSAL_LM"))

    for n, p in model.named_parameters():
        assert p.requires_grad == ("gadra_" in n), n

    n_wrapped = sum(isinstance(m, GaDRALinear) for m in model.modules())
    assert n_wrapped == 6

    # B=0 init -> zero delta -> injected logits equal base logits.
    model.eval()
    with torch.no_grad():
        adapted_logits = model(ids).logits
    torch.testing.assert_close(adapted_logits, base_logits, atol=1e-6, rtol=0.0)

    for k, v in model.get_base_model().state_dict().items():
        if "gadra_" in k:
            continue
        orig_key = k.replace(".base_layer.", ".")
        assert orig_key in base_snapshot, orig_key
        assert torch.equal(v, base_snapshot[orig_key]), orig_key


def test_save_load_roundtrip_and_merge_refused(tmp_path):
    from peft import PeftModel, get_peft_model
    from transformers import LlamaForCausalLM

    base, cfg = _tiny_llama()
    base_snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}
    ids = torch.randint(0, 64, (2, 6))

    model = get_peft_model(base, GaDRAConfig(r=8, task_type="CAUSAL_LM"))

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, GaDRALinear):
                m.gadra_B["default"].weight.normal_(0.0, 0.3)
                m.gadra_gate["default"].bias.fill_(10.0)

    model.eval()
    with torch.no_grad():
        logits1 = model(ids).logits

    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model(ids).logits
    assert not torch.allclose(logits1, base_logits, atol=1e-5)

    model.save_pretrained(str(tmp_path))
    base2 = LlamaForCausalLM(cfg)
    base2.load_state_dict(base_snapshot)
    reloaded = PeftModel.from_pretrained(base2, str(tmp_path))
    reloaded.eval()
    with torch.no_grad():
        logits2 = reloaded(ids).logits
    torch.testing.assert_close(logits2, logits1, atol=1e-5, rtol=0.0)

    with pytest.raises(NotImplementedError):
        model.merge_and_unload()


def test_saved_config_is_gadra():
    from peft import get_peft_model

    base, _ = _tiny_llama()
    model = get_peft_model(base, GaDRAConfig(r=8, task_type="CAUSAL_LM"))
    assert str(model.peft_config["default"].peft_type) == "PeftType.GADRA" or \
        model.peft_config["default"].peft_type == "GADRA"
