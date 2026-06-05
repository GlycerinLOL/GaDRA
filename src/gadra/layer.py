"""``GaDRALinear`` — a single ``nn.Linear`` wrapped with the GaDRA gated residual adapter.

Reproduces the released ``PeftBlock`` forward exactly (for the paper's ``alpha=1`` configs, bit-exact
in eval; controlled-RNG-equal in train):

    base_out = base_layer(x)                          # frozen
    delta    = alpha * gadra_B(gadra_A(dropout(x)))   # A: kaiming, B: zeros; no bias; no /r
    sig      = delta (mono) | cat(base_out, delta) (dual)
    z        = gadra_gate(dropout(sig))               # affine -> per-token scalar
    gamma    = compute_gamma(z, ...)                  # hard Gumbel-STE / soft softplus
    return base_out + gamma * delta

The two ``dropout`` calls (adapter input, then gate input) and the Gumbel draw happen in this order
to match the original RNG sequence. Non-mergeable: ``gamma`` is per-token and input-dependent.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .gate import compute_gamma


class GaDRALinear(nn.Module, BaseTunerLayer):
    # Trainable adapter submodules (all carry the ``gadra_`` prefix -> auto-saved by peft).
    adapter_layer_names: tuple[str, ...] = ("gadra_A", "gadra_B", "gadra_gate")
    # Per-adapter non-module hyperparameters.
    other_param_names: tuple[str, ...] = (
        "r",
        "lora_alpha",
        "lora_dropout",
        "router_conditioning",
        "gate",
        "gamma_threshold",
        "tau",
        "gate_bias_init",
    )

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        router_conditioning: str,
        gate: str,
        gamma_threshold: float = 0.5,
        tau: float = 1.0,
        gate_bias_init: Optional[float] = None,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self._disable_adapters = False
        self.merged_adapters: list[str] = []

        self.gadra_A = nn.ModuleDict({})
        self.gadra_B = nn.ModuleDict({})
        self.gadra_gate = nn.ModuleDict({})
        self.lora_dropout = nn.ModuleDict({})
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, float] = {}
        self.router_conditioning: dict[str, str] = {}
        self.gate: dict[str, str] = {}
        self.gamma_threshold: dict[str, float] = {}
        self.tau: dict[str, float] = {}
        self.gate_bias_init: dict[str, Optional[float]] = {}

        base = self.get_base_layer()
        self.in_features = base.in_features
        self.out_features = base.out_features

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            router_conditioning=router_conditioning,
            gate=gate,
            gamma_threshold=gamma_threshold,
            tau=tau,
            gate_bias_init=gate_bias_init,
            init_weights=init_weights,
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        router_conditioning: str,
        gate: str,
        gamma_threshold: float = 0.5,
        tau: float = 1.0,
        gate_bias_init: Optional[float] = None,
        init_weights: bool = True,
    ) -> None:
        if r <= 0:
            raise ValueError(f"r must be > 0, got: {r}")
        if router_conditioning not in ("dual", "mono"):
            raise ValueError(f"router_conditioning must be 'dual' or 'mono', got: {router_conditioning}")
        if gate not in ("hard", "soft"):
            raise ValueError(f"gate must be 'hard' or 'soft', got: {gate}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.router_conditioning[adapter_name] = router_conditioning
        self.gate[adapter_name] = gate
        self.gamma_threshold[adapter_name] = gamma_threshold
        self.tau[adapter_name] = tau
        self.gate_bias_init[adapter_name] = gate_bias_init

        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.gadra_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.gadra_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        gate_in_dim = 2 * self.out_features if router_conditioning == "dual" else self.out_features
        self.gadra_gate[adapter_name] = nn.Linear(gate_in_dim, 1, bias=True)

        if init_weights:
            self.reset_adapter_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_adapter_parameters(self, adapter_name: str) -> None:
        # LoRA-standard init: A ~ kaiming_uniform(a=sqrt(5)), B = 0 (so delta=0 at start).
        nn.init.kaiming_uniform_(self.gadra_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.gadra_B[adapter_name].weight)
        # Gate: default nn.Linear init (released-code behavior). gate_bias_init overrides b_g to a
        # constant when the user wants the paper's "gates start open" positive bias.
        bias_init = self.gate_bias_init[adapter_name]
        if bias_init is not None:
            nn.init.constant_(self.gadra_gate[adapter_name].bias, float(bias_init))

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError(
            "GaDRA is non-mergeable: gamma is a per-token, input-dependent gate, so the adapter "
            "cannot be folded into the base weight. Run inference with the adapter attached."
        )

    def unmerge(self) -> None:
        raise NotImplementedError("GaDRA is non-mergeable; there is nothing to unmerge.")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Mixed adapter batches (peft's per-sample adapter_names) are not supported.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError("GaDRA does not support mixed adapter batches (adapter_names).")

        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs)

        base_out = self.base_layer(x, *args, **kwargs)
        result_dtype = base_out.dtype
        result = base_out

        for active_adapter in self.active_adapters:
            if active_adapter not in self.gadra_A:
                continue

            lora_A = self.gadra_A[active_adapter]
            lora_B = self.gadra_B[active_adapter]
            gate_mod = self.gadra_gate[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            alpha = self.lora_alpha[active_adapter]
            conditioning = self.router_conditioning[active_adapter]
            gate_kind = self.gate[active_adapter]
            threshold = self.gamma_threshold[active_adapter]
            tau = self.tau[active_adapter]

            x_in = x.to(lora_A.weight.dtype)
            # draw #1: dropout on the adapter input.
            delta = alpha * lora_B(lora_A(dropout(x_in)))

            if conditioning == "dual":
                # Gate conditions on the frozen base output (NOT the running result), matching the
                # original per-module PeftBlock and keeping multi-adapter stacking order-independent.
                gate_input = torch.cat((base_out.to(delta.dtype), delta), dim=-1)
            else:  # mono
                gate_input = delta

            # draw #2: dropout on the gate input (same dropout module, separate draw).
            z = gate_mod(dropout(gate_input))
            # draw #3 (hard+train only): Gumbel sample inside compute_gamma.
            gamma = compute_gamma(z, gate=gate_kind, training=self.training, gamma_threshold=threshold, tau=tau)

            result = result + (gamma * delta).to(result_dtype)

        return result.to(result_dtype)

    def __repr__(self) -> str:
        return "gadra." + super().__repr__()
