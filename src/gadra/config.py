"""``GaDRAConfig`` — a flat, ``LoraConfig``-idiomatic configuration for the GaDRA tuner.

GaDRA = Gated Dual-conditioned Residual Adapter. The paper uses **uniform** settings across the
three MLP projections, so this config is flat (LoRA-style ``r`` / ``lora_alpha`` / ``lora_dropout``
/ ``target_modules``) plus the GaDRA gate fields, rather than a nested per-module map.

Output: ``y = y0 + gamma * delta`` with ``delta = lora_alpha * B(A(x))`` (no ``/r`` division; the
paper's ``alpha`` is the residual scale, with ``alpha = 1`` in all published runs). ``gamma`` is a
per-token scalar produced by a per-module affine gate. Non-mergeable by construction (``gamma`` is
input-dependent) — merging is refused in the layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType

from ._enum_shim import ensure_gadra_peft_type

# PeftType.GADRA must exist before it is referenced as a field default below.
ensure_gadra_peft_type()

#: Paper-faithful default targets: the three Llama/Qwen MLP projections (name-matched on any model).
DEFAULT_TARGET_MODULES = ["up_proj", "gate_proj", "down_proj"]

ROUTER_CONDITIONING_CHOICES = ("dual", "mono")
GATE_CHOICES = ("hard", "soft")

# Legacy nested-config fields that the converter ingests and drops (regularizers, analysis knobs,
# out-of-scope method variants). See docs/DESIGN.md §3 / §7.
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
        r: LoRA rank of the residual adapter (paper: 512).
        target_modules: Module name(s) to wrap (name-matched). Defaults to the three MLP projections.
        lora_alpha: Residual scale ``alpha`` in ``delta = alpha * B(A(x))`` (paper: 1). Applied
            directly, NOT divided by ``r`` (this differs from the standard LoRA ``alpha/r`` scaling).
        lora_dropout: Dropout applied to the adapter input and, separately, to the gate input.
        router_conditioning: ``"dual"`` (GaDRA, gate sees ``[y0; delta]``) or ``"mono"``
            (GaDRA-Mono, gate sees ``delta`` only).
        gate: ``"hard"`` (binary Gumbel-sigmoid STE in training; deterministic ``1[sigmoid(z)>thr]``
            at inference) or ``"soft"`` (continuous gate).
        gamma_threshold: Decision threshold for the hard gate.
        tau: Gumbel-sigmoid temperature (paper: 1.0, no annealing).
        gate_bias_init: Initial value for the gate bias ``b_g``. ``None`` reproduces the released
            code (default ``nn.Linear`` init). A positive float matches the paper-text description
            ("gates start open"); the released runs used the default init.
        rank_pattern / alpha_pattern: Optional peft-standard per-module overrides for heterogeneity
            (unused by the paper's uniform config).
        modules_to_save: Standard peft passthrough.
    """

    r: int = field(default=512, metadata={"help": "LoRA rank of the residual adapter."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "Module name(s) to wrap; defaults to the three MLP projections."}
    )
    lora_alpha: float = field(default=1.0, metadata={"help": "Residual scale alpha (delta = alpha * B @ A @ x)."})
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GADRA

        if self.target_modules is None:
            self.target_modules = list(DEFAULT_TARGET_MODULES)

        if self.router_conditioning not in ROUTER_CONDITIONING_CHOICES:
            raise ValueError(
                f"router_conditioning must be one of {ROUTER_CONDITIONING_CHOICES}, got: {self.router_conditioning}"
            )
        if self.gate not in GATE_CHOICES:
            raise ValueError(f"gate must be one of {GATE_CHOICES}, got: {self.gate}")
        if self.r <= 0:
            raise ValueError(f"r must be > 0, got: {self.r}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got: {self.tau}")

    @classmethod
    def from_legacy_peft_config(cls, legacy: dict) -> "GaDRAConfig":
        """Build a flat ``GaDRAConfig`` from a legacy nested ``peft_config.json`` dict.

        Maps ``router_type`` (MA→dual, UniPELT→mono) and ``gamma_hard_masking`` (Gumbel→hard,
        None→soft), asserts the three target modules are uniform and in scope, and drops all
        regularizer / analysis / out-of-scope-variant fields. Raises for configs outside the GaDRA
        paper scope (svd_minor, Add/relu, STE, mlp_external, partial layer DSL).
        """
        modules = legacy.get("target_modules")
        if not isinstance(modules, dict) or not modules:
            raise ValueError("legacy config must carry a non-empty nested 'target_modules' dict")

        # Legacy conversion is paper-faithful: only the three MLP projections are convertible.
        # (The runtime tuner stays generic and can wrap any Linear, but attention adapters, whole-MLP
        # wrappers, etc. have a different weight layout / are out of the published method's scope.)
        _convertible_modules = {"up_proj", "down_proj", "gate_proj"}
        out_of_scope = [name for name in modules if name not in _convertible_modules]
        if out_of_scope:
            raise ValueError(
                f"target modules {out_of_scope} are out of GaDRA conversion scope; only "
                f"{sorted(_convertible_modules)} (the paper's MLP projections) are convertible"
            )

        # Reject out-of-scope partial-layer selection (the gate budget is learned, not hand-set).
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

        # GaDRAConfig applies alpha to delta (delta = alpha * B@A) and is bit-exact to the original
        # only when the legacy output_scalar (which scaled the gate, not the adapter) is 1.0.
        output_scalar = float(first.get("output_scalar", 1.0))
        if output_scalar != 1.0:
            raise ValueError(
                f"output_scalar={output_scalar} is out of conversion scope: the flat layer applies alpha "
                "to delta and reproduces the original only for output_scalar == 1.0 (the paper setting)."
            )

        return cls(
            r=int(first.get("peft_rank", 512)),
            target_modules=list(modules.keys()),
            lora_alpha=output_scalar,
            lora_dropout=float(first.get("dropout", 0.05)),
            router_conditioning=conditioning,
            gate=gate,
            gamma_threshold=float(first.get("gamma_threshold", 0.5)),
            tau=1.0,
            gate_bias_init=None,
        )
