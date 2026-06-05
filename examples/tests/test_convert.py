"""P3 gate — legacy -> peft-native checkpoint conversion.

Builds a REAL legacy checkpoint with the research repo's original ``PeftPatcher`` (imported from the
source repo), converts it, loads the result through the standard ``PeftModel.from_pretrained``, and
asserts every adapter weight transferred bit-for-bit (direct param equality — independent of any
model forward). Also checks bijection, config mapping, and out-of-scope rejection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

import gadra  # noqa: F401  (registers the "gadra" method)
from gadra import GaDRAConfig
from examples.convert import convert_checkpoint

_SOURCE_REPO = os.environ.get("GADRA_SOURCE_REPO") or str(Path(__file__).resolve().parents[3])
if _SOURCE_REPO not in sys.path:
    sys.path.insert(0, _SOURCE_REPO)

_MLP_MODULES = ["up_proj", "down_proj", "gate_proj"]


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
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), cfg


def _to_model_param_name(saved_key: str) -> str:
    """Saved safetensors key (no adapter name) -> the param name peft uses after load (.default)."""
    head, param = saved_key.rsplit(".", 1)
    return f"{head}.default.{param}"


def _build_legacy_checkpoint(out_dir: Path, freeze: bool = True):
    """Patch a tiny model with the ORIGINAL implementation and save a legacy checkpoint."""
    try:
        from peft_config import ModulePeftConfig, PeftConfig as LegacyPeftConfig
        from peft_module import PeftPatcher
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"original implementation not importable: {exc}")

    base, _cfg = _tiny_llama()
    legacy_cfg = LegacyPeftConfig(
        peft_block_name="LoRA",
        target_modules={
            mod: ModulePeftConfig(
                peft_rank=8,
                peft_type="lora",
                router_type="MA",  # dual
                gamma_hard_masking="Gumbel",  # hard
            )
            for mod in _MLP_MODULES
        },
    )
    patcher = PeftPatcher(base, legacy_cfg)
    model_old = patcher.patch_model_with_peft()

    # Real GaDRA training freezes the base and trains only the adapters, so the saved checkpoint
    # contains only peft params. Mirror that here (freeze=False keeps base params trainable to
    # exercise the converter's extra-trainable guard).
    for n, p in model_old.named_parameters():
        p.requires_grad_(("peft_blocks" in n) or (not freeze))

    # Make every adapter param distinct & non-zero so the equality check is meaningful.
    from peft_model import PeftBlock

    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for module in model_old.modules():
            if isinstance(module, PeftBlock):
                for p in module.parameters():
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.3)

    patcher.save_peft_adapter(str(out_dir))
    return out_dir / "LoRA"


def test_convert_transfers_every_weight_and_loads_via_peft(tmp_path):
    legacy_dir = _build_legacy_checkpoint(tmp_path / "old")
    old_state = torch.load(legacy_dir / "peft_model.bin", map_location="cpu", weights_only=True)

    out_dir = tmp_path / "new"
    manifest = convert_checkpoint(tmp_path / "old", out_dir)

    # 3 projections x 2 layers x {A.weight, B.weight, gate.weight, gate.bias} = 24 params.
    assert manifest["num_params"] == len(old_state) == 24
    assert manifest["config"] == {
        "r": 8,
        "router_conditioning": "dual",
        "gate": "hard",
        "lora_alpha": 1.0,
        "lora_dropout": 0.05,
        "target_modules": ["up_proj", "down_proj", "gate_proj"],
    }
    saved = json.loads((out_dir / "adapter_config.json").read_text())
    assert saved["peft_type"] == "GADRA"

    # Load through the real peft API and verify every weight transferred exactly.
    from peft import PeftModel

    base2, _ = _tiny_llama()
    m2 = PeftModel.from_pretrained(base2, str(out_dir))
    params = dict(m2.named_parameters())

    for old_key, new_key in manifest["key_map"].items():
        model_name = _to_model_param_name(new_key)
        assert model_name in params, model_name
        assert torch.equal(params[model_name], old_state[old_key]), old_key

    assert m2.peft_config["default"].router_conditioning == "dual"
    assert m2.peft_config["default"].gate == "hard"
    assert m2.peft_config["default"].r == 8


def test_convert_rejects_extra_trainable_by_default(tmp_path):
    # Unfrozen base -> legacy save also stores base params -> conversion must refuse by default.
    legacy_dir = _build_legacy_checkpoint(tmp_path / "old", freeze=False)
    assert legacy_dir.exists()
    with pytest.raises(ValueError):
        convert_checkpoint(tmp_path / "old", tmp_path / "new")
    # ...but allow_extra_trainable drops them and succeeds.
    manifest = convert_checkpoint(tmp_path / "old", tmp_path / "new2", allow_extra_trainable=True)
    assert manifest["num_params"] == 24
    assert len(manifest["skipped_keys"]) > 0


def test_convert_rejects_out_of_scope(tmp_path):
    legacy = tmp_path / "ste" / "LoRA"
    legacy.mkdir(parents=True)
    ste_cfg = {
        "peft_block_name": "LoRA",
        "gate_layers": None,
        "peft_layers": "all",
        "target_modules": {
            "up_proj": {
                "peft_rank": 512,
                "peft_type": "lora",
                "init_method": "default",
                "activation": "none",
                "router_type": "MA",
                "router_input": "default",
                "output_scalar": 1.0,
                "max_gating": False,
                "gamma_hard_masking": "STE",  # out of scope
            }
        },
    }
    (legacy / "peft_config.json").write_text(json.dumps(ste_cfg))
    torch.save({}, legacy / "peft_model.bin")

    with pytest.raises(ValueError):
        convert_checkpoint(tmp_path / "ste", tmp_path / "ste_out")


def test_from_legacy_used_by_converter_is_consistent():
    # Sanity: the converter's config mapping matches GaDRAConfig.from_legacy_peft_config directly.
    legacy_path = Path(_SOURCE_REPO) / "configs" / "peft_config.json"
    if not legacy_path.exists():
        pytest.skip("canonical legacy config not present")
    cfg = GaDRAConfig.from_legacy_peft_config(json.loads(legacy_path.read_text()))
    assert cfg.router_conditioning == "dual"
    assert cfg.gate == "hard"
