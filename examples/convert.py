"""One-time batch converter: legacy GaDRA checkpoint -> peft-native adapter.

Legacy layout (written by the research repo's ``PeftPatcher.save_peft_adapter``):

    <ckpt>/LoRA/peft_config.json      # nested per-module config
    <ckpt>/LoRA/peft_model.bin        # state_dict keyed by full patched-model param paths,
                                      #   e.g. model.layers.0.mlp.peft_blocks.up_proj.adapter.0.weight

New layout (standard peft):

    <out>/adapter_config.json         # flat GaDRAConfig (peft_type=GADRA)
    <out>/adapter_model.safetensors   # keys base_model.model.<path>.<mod>.gadra_{A,B,gate}.{weight,bias}
                                      #   (no adapter name — peft re-inserts ".default" on load)

Key map (per decoder layer, per MLP projection ``<mod>`` in {up_proj, down_proj, gate_proj}):

    ...mlp.peft_blocks.<mod>.adapter.0.weight  ->  base_model.model.<...>.mlp.<mod>.gadra_A.weight
    ...mlp.peft_blocks.<mod>.adapter.1.weight  ->  base_model.model.<...>.mlp.<mod>.gadra_B.weight
    ...mlp.peft_blocks.<mod>.gating.0.weight   ->  base_model.model.<...>.mlp.<mod>.gadra_gate.weight
    ...mlp.peft_blocks.<mod>.gating.0.bias     ->  base_model.model.<...>.mlp.<mod>.gadra_gate.bias

Originals are read-only; the converter only writes under ``out_dir``. Out-of-scope checkpoints
(whole-MLP ``mlp_external``, attention targets, ``svd_minor``/MiLoRA, ``STE``) are rejected at the
config stage by :meth:`GaDRAConfig.from_legacy_peft_config`, or here for unrecognized weight keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from gadra.config import GaDRAConfig

# old key -> (path-up-to-and-including .mlp, module, adapter.0|adapter.1|gating.0, weight|bias)
_OLD_KEY_RE = re.compile(
    r"^(?P<path>.+\.mlp)\.peft_blocks\.(?P<mod>[^.]+)\.(?P<sub>adapter\.[01]|gating\.0)\.(?P<param>weight|bias)$"
)
_SUB_TO_GADRA = {"adapter.0": "gadra_A", "adapter.1": "gadra_B", "gating.0": "gadra_gate"}
_IN_SCOPE_MODULES = frozenset({"up_proj", "down_proj", "gate_proj"})


def _resolve_legacy_dir(old_dir: str | Path) -> Path:
    """Find the directory holding both peft_config.json and peft_model.bin."""
    old = Path(old_dir)
    candidates = [old, old / "LoRA"]
    for cand in candidates:
        if (cand / "peft_config.json").is_file() and (cand / "peft_model.bin").is_file():
            return cand
    for cfg in sorted(old.rglob("peft_config.json")):
        if (cfg.parent / "peft_model.bin").is_file():
            return cfg.parent
    raise FileNotFoundError(
        f"Could not find a legacy checkpoint (peft_config.json + peft_model.bin) under {old}"
    )


def _remap_key(old_key: str) -> str:
    m = _OLD_KEY_RE.match(old_key)
    if m is None:
        raise ValueError(
            f"Out-of-scope adapter key (only MLP up/down/gate projections are supported): {old_key!r}. "
            "Whole-MLP and attention adapters are out of scope."
        )
    mod = m["mod"]
    if mod not in _IN_SCOPE_MODULES:
        raise ValueError(
            f"Out-of-scope target module {mod!r} in key {old_key!r}; expected one of {sorted(_IN_SCOPE_MODULES)}."
        )
    gadra_layer = _SUB_TO_GADRA[m["sub"]]
    # No adapter name in the saved file — peft re-inserts ".default" on load.
    return f"base_model.model.{m['path']}.{mod}.{gadra_layer}.{m['param']}"


def _assert_adapter_shapes(state: dict[str, torch.Tensor], cfg: GaDRAConfig) -> None:
    """Per-module shape validation: rank, completeness, and gate width vs router_conditioning."""
    from collections import defaultdict

    groups: dict[str, dict[str, tuple]] = defaultdict(dict)
    for key, tensor in state.items():
        prefix, layer, param = key.rsplit(".", 2)  # ...mlp.<mod> | gadra_X | weight|bias
        groups[prefix][f"{layer}.{param}"] = tuple(tensor.shape)

    dual = cfg.router_conditioning == "dual"
    for prefix, g in groups.items():
        a, b = g.get("gadra_A.weight"), g.get("gadra_B.weight")
        gw, gb = g.get("gadra_gate.weight"), g.get("gadra_gate.bias")
        if not (a and b and gw and gb):
            raise ValueError(f"incomplete adapter at {prefix}: have {sorted(g)}")
        if a[0] != cfg.r or b[1] != cfg.r:
            raise ValueError(f"rank mismatch at {prefix}: A{a}, B{b}, expected r={cfg.r}")
        out_features = b[0]
        expected_gate_in = 2 * out_features if dual else out_features
        if gw != (1, expected_gate_in):
            raise ValueError(
                f"gate weight {gw} at {prefix} incompatible with router_conditioning="
                f"{cfg.router_conditioning!r} (expected (1, {expected_gate_in}))"
            )
        if gb != (1,):
            raise ValueError(f"gate bias {gb} at {prefix} != (1,)")


def convert_checkpoint(old_dir: str | Path, out_dir: str | Path, *, allow_extra_trainable: bool = False) -> dict:
    """Convert a legacy GaDRA checkpoint to the peft-native format.

    Returns a manifest dict (source, dest, parameter count, mapped config, full key map). Writes
    ``adapter_config.json``, ``adapter_model.safetensors`` and ``convert_manifest.json`` under
    ``out_dir``. Asserts a bijective key remap with preserved shapes.

    A frozen-base GaDRA checkpoint contains only adapter params. If the legacy checkpoint also holds
    non-adapter trainable params (embeddings / lm_head / norms / modules-to-save), conversion raises
    unless ``allow_extra_trainable=True`` (in which case they are dropped and recorded in the manifest).
    """
    src = _resolve_legacy_dir(old_dir)

    legacy_cfg = json.loads((src / "peft_config.json").read_text(encoding="utf-8"))
    gadra_cfg = GaDRAConfig.from_legacy_peft_config(legacy_cfg)

    old_state = torch.load(src / "peft_model.bin", map_location="cpu", weights_only=True)

    new_state: dict[str, torch.Tensor] = {}
    key_map: dict[str, str] = {}
    skipped_keys: list[str] = []
    for old_key, tensor in old_state.items():
        # Pure base params (e.g. embed_tokens / lm_head / norms left trainable) are not adapter
        # weights — skip them. Adapter params always live under a ".peft_blocks." container.
        if ".peft_blocks." not in old_key:
            skipped_keys.append(old_key)
            continue
        new_key = _remap_key(old_key)  # raises for out-of-scope adapters (attention / whole-MLP)
        if new_key in new_state:
            raise ValueError(f"Key collision while converting: {new_key!r}")
        new_state[new_key] = tensor.detach().cpu().contiguous()
        key_map[old_key] = new_key

    if not new_state:
        raise ValueError(f"No GaDRA adapter params found under {src}; nothing to convert.")
    if skipped_keys and not allow_extra_trainable:
        raise ValueError(
            f"legacy checkpoint has {len(skipped_keys)} non-adapter trainable param(s) "
            f"(e.g. {skipped_keys[:3]}); a frozen-base GaDRA checkpoint should have none. "
            "Pass allow_extra_trainable=True to drop them."
        )
    # Bijection: each converted source key maps to a unique destination key.
    if len(key_map) != len(new_state):
        raise AssertionError("non-bijective key remap detected")
    if len(new_state) + len(skipped_keys) != len(old_state):
        raise AssertionError("key accounting mismatch (converted + skipped != total)")
    # Per-module shape integrity (rank + gate width vs router_conditioning).
    _assert_adapter_shapes(new_state, gadra_cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gadra_cfg.save_pretrained(str(out))
    save_file(new_state, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})

    manifest = {
        "source": str(src),
        "dest": str(out),
        "num_params": len(new_state),
        "config": {
            "r": gadra_cfg.r,
            "router_conditioning": gadra_cfg.router_conditioning,
            "gate": gadra_cfg.gate,
            "lora_alpha": gadra_cfg.lora_alpha,
            "lora_dropout": gadra_cfg.lora_dropout,
            "target_modules": list(gadra_cfg.target_modules),
        },
        "key_map": key_map,
        "skipped_keys": skipped_keys,
    }
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
