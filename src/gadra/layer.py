"""``GaDRALinear`` — an ``nn.Linear`` wrapped with the GaDRA gated residual adapter.

    base_out = base_layer(x)
    delta    = (lora_alpha / r) * gadra_B(gadra_A(dropout(x)))
    gate_in  = delta (mono) | cat(base_out, delta) (dual)
    gamma    = compute_gamma(gadra_gate(dropout(gate_in)), ...)
    return base_out + gamma * delta

Non-mergeable: ``gamma`` is a per-token, input-dependent gate.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .gate import compute_gamma


class GaDRALinear(nn.Module, BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("gadra_A", "gadra_B", "gadra_gate")
    other_param_names: tuple[str, ...] = (
        "r",
        "lora_alpha",
        "scaling",
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
        init_lora_weights: bool = True,
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
        self.scaling: dict[str, float] = {}  # lora_alpha / r
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
            init_lora_weights=init_lora_weights,
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
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        if r <= 0:
            raise ValueError(f"r must be > 0, got: {r}")
        if router_conditioning not in ("dual", "mono"):
            raise ValueError(f"router_conditioning must be 'dual' or 'mono', got: {router_conditioning}")
        if gate not in ("hard", "soft"):
            raise ValueError(f"gate must be 'hard' or 'soft', got: {gate}")
        if not isinstance(init_lora_weights, bool):
            raise ValueError(
                "GaDRA supports init_lora_weights as a bool only (True/False); peft's string variants "
                f"('gaussian'/'pissa'/...) are out of scope, got: {init_lora_weights!r}"
            )
        # reject unsupported peft LoRA options instead of silently ignoring them
        for opt in ("use_rslora", "use_dora", "use_qalora", "lora_bias"):
            if kwargs.get(opt):
                raise ValueError(f"GaDRA does not support the peft LoRA option {opt!r}.")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
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

        if init_lora_weights:
            self.reset_adapter_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_adapter_parameters(self, adapter_name: str) -> None:
        # A: kaiming_uniform; B: zeros (delta = 0 at init)
        nn.init.kaiming_uniform_(self.gadra_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.gadra_B[adapter_name].weight)
        # optional constant gate-bias init
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
            scaling = self.scaling[active_adapter]
            conditioning = self.router_conditioning[active_adapter]
            gate_kind = self.gate[active_adapter]
            threshold = self.gamma_threshold[active_adapter]
            tau = self.tau[active_adapter]

            x_in = x.to(lora_A.weight.dtype)
            delta = scaling * lora_B(lora_A(dropout(x_in)))

            if conditioning == "dual":
                gate_input = torch.cat((base_out.to(delta.dtype), delta), dim=-1)
            else:  # mono
                gate_input = delta

            z = gate_mod(dropout(gate_input))
            gamma = compute_gamma(z, gate=gate_kind, training=self.training, gamma_threshold=threshold, tau=tau)

            result = result + (gamma * delta).to(result_dtype)

        return result.to(result_dtype)

    def __repr__(self) -> str:
        return "gadra." + super().__repr__()
