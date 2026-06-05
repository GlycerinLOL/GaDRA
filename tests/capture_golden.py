"""P0 — freeze the golden parity baseline from the *original* GaDRA ``PeftBlock``.

Runs the released ``peft_model.PeftBlock`` forward (imported verbatim from the source research repo)
for all four shipped variants — dual/mono router conditioning x hard/soft gate — on a tiny CPU
tensor, and saves weights + inputs + outputs to ``tests/_golden/``. The P2 ``GaDRALinear`` must
reproduce these:

* **eval mode** is the BIT-EXACT target (dropout off, Gumbel deterministic, softplus deterministic);
* **train mode** is captured under a fixed pre-forward seed as the within-tolerance reference (its
  RNG order — 2 dropout draws + 1 Gumbel ``rand_like`` per gated module — may differ across stacks).

Run once on the execution host (or any CPU box with the env):

    GADRA_SOURCE_REPO=/path/to/Adapter_Research \
        python GaDRA/tests/capture_golden.py

Output is deterministic given the seeds below, so re-running reproduces identical goldens.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# --- locate and import the original implementation -------------------------------------------------
_SOURCE_REPO = os.environ.get("GADRA_SOURCE_REPO")
if _SOURCE_REPO is None:
    # tests/capture_golden.py -> GaDRA -> <source repo>
    _SOURCE_REPO = str(Path(__file__).resolve().parents[2])
if _SOURCE_REPO not in sys.path:
    sys.path.insert(0, _SOURCE_REPO)

from peft_config import ModulePeftConfig  # noqa: E402
from peft_model import PeftBlock  # noqa: E402

_GOLDEN_DIR = Path(__file__).resolve().parent / "_golden"

# Tiny, gate-exercising dimensions. D_out must equal the adapter output dim so the dual gate's
# concat([y0, delta]) and mono gate's [delta] have consistent shapes.
_IN_FEATURES = 16
_OUT_FEATURES = 24
_RANK = 8
_BATCH = 2
_SEQ = 5
_DROPOUT = 0.05
_GAMMA_THRESHOLD = 0.5
_TRAIN_SEED = 4242

# (variant_name, router_type, gamma_hard_masking, weight_seed) — router_type MA=dual / UniPELT=mono;
# gamma_hard_masking Gumbel=hard / None=soft. Fixed per-variant seeds keep captures reproducible
# (Python's hash() is process-randomized and must not be used for seeding).
_VARIANTS = [
    ("dual_hard", "MA", "Gumbel", 1001),
    ("dual_soft", "MA", None, 1002),
    ("mono_hard", "UniPELT", "Gumbel", 1003),
    ("mono_soft", "UniPELT", None, 1004),
]


def _build_block(router_type: str, gamma_hard_masking) -> PeftBlock:
    cfg = ModulePeftConfig(
        peft_rank=_RANK,
        peft_type="lora",
        init_method="default",
        router_type=router_type,
        router_input="default",
        activation="none",
        max_gating=False,
        output_scalar=1.0,
        dropout=_DROPOUT,
        gamma_hard_masking=gamma_hard_masking,
        gamma_threshold=_GAMMA_THRESHOLD,
    )
    # compute_cr=False: analysis/diagnostics are out of scope and would add no-op branches.
    return PeftBlock(_IN_FEATURES, _OUT_FEATURES, cfg, compute_cr=False)


def _randomize(block: PeftBlock, seed: int) -> None:
    """Fill every parameter with N(0, 0.5) so the adapter delta is non-zero (B inits to zeros) and
    the gate produces a non-trivial mix of 0/1 across tokens. Values are arbitrary — parity copies
    this exact state_dict into the new layer."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in block.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.5)


def capture_variant(name: str, router_type: str, gamma_hard_masking, weight_seed: int) -> dict:
    block = _build_block(router_type, gamma_hard_masking)
    _randomize(block, seed=weight_seed)

    in_gen = torch.Generator().manual_seed(7)
    x = torch.randn(_BATCH, _SEQ, _IN_FEATURES, generator=in_gen)
    y0 = torch.randn(_BATCH, _SEQ, _OUT_FEATURES, generator=in_gen)

    # --- eval mode: bit-exact target ---
    block.eval()
    with torch.no_grad():
        eval_out = block(x, y0)

    # --- train mode: controlled-RNG within-tolerance reference ---
    block.train()
    torch.manual_seed(_TRAIN_SEED)
    train_out = block(x, y0)  # grad-enabled; detach before saving

    record = {
        "name": name,
        "router_type": router_type,
        "gamma_hard_masking": gamma_hard_masking,
        "dims": {
            "in_features": _IN_FEATURES,
            "out_features": _OUT_FEATURES,
            "rank": _RANK,
            "batch": _BATCH,
            "seq": _SEQ,
        },
        "dropout": _DROPOUT,
        "gamma_threshold": _GAMMA_THRESHOLD,
        # PeftBlock key layout (activation='none'): adapter.0=A, adapter.1=B, gating.0=affine gate.
        "key_map": {"adapter.0.weight": "gadra_A", "adapter.1.weight": "gadra_B", "gating.0": "gadra_gate"},
        "state_dict": {k: v.detach().cpu() for k, v in block.state_dict().items()},
        "x": x,
        "y0": y0,
        "eval_output": eval_out.output.detach().cpu(),
        "eval_gamma": eval_out.gamma.detach().cpu() if eval_out.gamma is not None else None,
        "train_seed": _TRAIN_SEED,
        "train_output": train_out.output.detach().cpu(),
        "train_gamma": train_out.gamma.detach().cpu() if train_out.gamma is not None else None,
    }
    return record


def main() -> None:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, router_type, gamma_hard_masking, weight_seed in _VARIANTS:
        record = capture_variant(name, router_type, gamma_hard_masking, weight_seed)
        out_path = _GOLDEN_DIR / f"golden_{name}.pt"
        torch.save(record, out_path)
        manifest.append(
            {
                "name": name,
                "file": out_path.name,
                "router_type": router_type,
                "gamma_hard_masking": gamma_hard_masking,
                "state_dict_keys": list(record["state_dict"].keys()),
                "eval_output_shape": list(record["eval_output"].shape),
            }
        )
        print(f"[golden] {name}: saved {out_path.name} ({len(record['state_dict'])} params)")

    with open(_GOLDEN_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[golden] manifest written; {len(manifest)} variants under {_GOLDEN_DIR}")


if __name__ == "__main__":
    main()
