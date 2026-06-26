"""``GaDRAConfig`` — a flat, ``LoraConfig``-idiomatic configuration for the GaDRA tuner.

GaDRA = Gated Dual-conditioned Residual Adapter. Output: ``y = y0 + gamma * delta`` with
``delta = (lora_alpha / r) * B(A(x))`` and ``gamma`` a per-token scalar from a per-module affine
gate. Non-mergeable (``gamma`` is input-dependent).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType

from ._enum_shim import ensure_gadra_peft_type

ensure_gadra_peft_type()

#: Default targets: the three MLP projections (name-matched on any model).
DEFAULT_TARGET_MODULES = ["up_proj", "gate_proj", "down_proj"]

ROUTER_CONDITIONING_CHOICES = ("dual", "mono")
GATE_CHOICES = ("hard", "soft", "none")

# Legacy nested-config fields the converter ingests and drops.
_LEGACY_TOP_LEVEL_DROP = frozenset(
    {
        "peft_block_name",
        "lora_weight_regularization",
        "lora_reg_lambda",
        "gamma_target_loss_enabled",
        "gamma_target_value",
        "gamma_target_loss_coef",
        "gamma_target_loss_type",
        "cr_loss_enabled",
        "cr_loss_threshold",
        "cr_loss_coef",
        "cr_loss_type",
        "r2_loss_enabled",
        "r2_loss_threshold",
        "r2_loss_coef",
        "r2_loss_type",
        "gate_layers",
        "peft_layers",
    }
)


@dataclass
class GaDRAConfig(PeftConfig):
    """Configuration for the GaDRA PEFT method.

    Args:
        r: LoRA rank of the residual adapter.
        target_modules: Module name(s) to wrap (name-matched). Defaults to the three MLP projections.
        lora_alpha: LoRA scaling numerator; ``delta`` is scaled by ``lora_alpha / r``. Defaults to
            ``r`` (effective scale 1.0).
        lora_dropout: Dropout applied to the adapter input and, separately, to the gate input.
        router_conditioning: ``"dual"`` (gate sees ``[y0; delta]``) or ``"mono"`` (gate sees ``delta``).
        gate: ``"hard"`` (Gumbel-sigmoid STE), ``"soft"`` (continuous), or ``"none"`` (ungated,
            ``gamma=1``; block-parallel LoRA baseline).
        gamma_threshold: Decision threshold for the hard gate.
        tau: Gumbel-sigmoid temperature.
        gate_bias_init: Initial value for the gate bias; ``None`` = default ``nn.Linear`` init.
        rank_pattern / alpha_pattern: Optional peft-standard per-module overrides.
        modules_to_save: Extra modules to set trainable and save.
        init_lora_weights: ``True`` = A kaiming, B zeros (delta 0 at start); ``False`` = default
            ``nn.Linear`` init (random B). Bool only.
        exclude_modules: Module name(s) to exclude from ``target_modules``.
        parallel_modules: Module name(s) wrapped as a whole-block parallel adapter (``A: H->r``,
            ``B: r->H``, gated, added parallel to the block output) — for MLP/MoE blocks. Non-mergeable.

    ``target_modules`` is kept as an ordered ``list``; ``layers_to_transform`` / ``layers_pattern``
    are unsupported.
    """

    r: int = field(default=512, metadata={"help": "LoRA rank of the residual adapter."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "Module name(s) to wrap; defaults to the three MLP projections."}
    )
    lora_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA scaling numerator; delta scaled by lora_alpha/r. None -> r (effective scale 1.0)."},
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout on adapter input and gate input."})
    router_conditioning: str = field(
        default="dual", metadata={"help": "'dual' (GaDRA) or 'mono' (GaDRA-Mono)."}
    )
    gate: str = field(default="hard", metadata={"help": "'hard' (Gumbel-STE) or 'soft' (continuous)."})
    gamma_threshold: float = field(default=0.5, metadata={"help": "Decision threshold for the hard gate."})
    tau: float = field(default=1.0, metadata={"help": "Gumbel-sigmoid temperature."})
    gate_bias_init: Optional[float] = field(
        default=None, metadata={"help": "Gate bias init; None = default nn.Linear init (released-code behavior)."}
    )
    rank_pattern: Optional[dict] = field(default_factory=dict, metadata={"help": "Per-module rank overrides."})
    alpha_pattern: Optional[dict] = field(default_factory=dict, metadata={"help": "Per-module alpha overrides."})
    modules_to_save: Optional[list[str]] = field(
        default=None, metadata={"help": "List of modules (besides adapters) to set trainable and save."}
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "True=LoRA-standard init (A kaiming, B zeros, delta=0 at start); "
                          "False=default nn.Linear init (random B). Bool only (no gaussian/pissa)."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "peft-standard module name(s) to exclude from target_modules."}
    )
    parallel_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module name(s) to wrap as a whole-block parallel adapter (e.g. ['mlp'] for an "
                          "MLP/MoE block): A: H->r, B: r->H, gated, added parallel to the block output. "
                          "Non-mergeable. When None, any matched non-Linear target is auto-wrapped block-parallel."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GADRA

        if self.target_modules is None:
            # block-parallel requests target the block via parallel_modules, not the per-projection default
            self.target_modules = [] if self.parallel_modules else list(DEFAULT_TARGET_MODULES)

        if self.router_conditioning not in ROUTER_CONDITIONING_CHOICES:
            raise ValueError(
                f"router_conditioning must be one of {ROUTER_CONDITIONING_CHOICES}, got: {self.router_conditioning}"
            )
        if self.gate not in GATE_CHOICES:
            raise ValueError(f"gate must be one of {GATE_CHOICES}, got: {self.gate}")
        if self.r <= 0:
            raise ValueError(f"r must be > 0, got: {self.r}")
        if self.lora_alpha is None:
            self.lora_alpha = float(self.r)
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got: {self.tau}")

    @classmethod
    def from_legacy_peft_config(cls, legacy: dict) -> "GaDRAConfig":
        """Build a flat ``GaDRAConfig`` from a legacy nested ``peft_config.json`` dict.

        Maps ``router_type`` (MA→dual, UniPELT→mono) and ``gamma_hard_masking`` (Gumbel→hard,
        None→soft); requires uniform, in-scope target modules and raises for out-of-scope configs.
        """
        modules = legacy.get("target_modules")
        if not isinstance(modules, dict) or not modules:
            raise ValueError("legacy config must carry a non-empty nested 'target_modules' dict")

        # warn on any legacy top-level key that is neither consumed nor in the known-drop set
        unexpected = set(legacy) - {"target_modules"} - _LEGACY_TOP_LEVEL_DROP
        if unexpected:
            warnings.warn(
                f"from_legacy_peft_config: ignoring unrecognized legacy top-level key(s) {sorted(unexpected)}",
                stacklevel=2,
            )

        # only the three MLP projections are convertible
        _convertible_modules = {"up_proj", "down_proj", "gate_proj"}
        out_of_scope = [name for name in modules if name not in _convertible_modules]
        if out_of_scope:
            raise ValueError(
                f"target modules {out_of_scope} are out of GaDRA conversion scope; only "
                f"{sorted(_convertible_modules)} (the paper's MLP projections) are convertible"
            )

        # partial-layer selection is out of scope
        for key in ("gate_layers", "peft_layers"):
            val = legacy.get(key)
            if val not in (None, "all"):
                raise ValueError(f"partial '{key}'={val!r} is out of GaDRA scope (per-layer budget is learned)")

        module_specs = list(modules.items())
        first_name, first = module_specs[0]
        for name, spec in module_specs:
            if spec != first:
                raise ValueError(
                    f"heterogeneous target_modules are out of scope: '{name}' differs from '{first_name}'"
                )

        if first.get("peft_type", "lora") != "lora":
            raise ValueError(f"only peft_type='lora' is in scope, got: {first.get('peft_type')}")
        if first.get("init_method", "default") != "default":
            raise ValueError(f"init_method must be 'default' (svd_minor/MiLoRA out of scope), got: {first.get('init_method')}")
        if first.get("router_input", "default") != "default":
            raise ValueError(f"router_input must be 'default', got: {first.get('router_input')}")
        if first.get("activation") not in (None, "none"):
            raise ValueError(f"activation must be 'none' for LoRA, got: {first.get('activation')}")
        if first.get("max_gating", False) not in (False, None, 0):
            raise ValueError(f"max_gating must be falsy (out of scope), got: {first.get('max_gating')}")

        router_type = first.get("router_type", "fixed")
        conditioning = {"MA": "dual", "UniPELT": "mono"}.get(router_type)
        if conditioning is None:
            raise ValueError(f"router_type {router_type!r} is out of scope (expected MA or UniPELT)")

        hard_masking = first.get("gamma_hard_masking")
        gate = {"Gumbel": "hard", None: "soft"}.get(hard_masking)
        if gate is None:
            raise ValueError(f"gamma_hard_masking {hard_masking!r} is out of scope (expected 'Gumbel' or null)")

        output_scalar = float(first.get("output_scalar", 1.0))
        if output_scalar != 1.0:
            raise ValueError(
                f"output_scalar={output_scalar} is out of conversion scope: the flat layer applies alpha "
                "to delta and reproduces the original only for output_scalar == 1.0 (the paper setting)."
            )

        peft_rank = int(first.get("peft_rank", 512))
        return cls(
            r=peft_rank,
            target_modules=list(modules.keys()),
            lora_alpha=output_scalar * peft_rank,
            lora_dropout=float(first.get("dropout", 0.05)),
            router_conditioning=conditioning,
            gate=gate,
            gamma_threshold=float(first.get("gamma_threshold", 0.5)),
            tau=1.0,
            gate_bias_init=None,
        )
