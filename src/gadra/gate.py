"""Gate functions for GaDRA.

``compute_gamma`` produces the per-token gate value for the two gate types:

* **hard**: training draws a Gumbel-sigmoid STE binary sample; inference is the deterministic
  threshold ``1[sigmoid(z) > threshold]``.
* **soft**: continuous ``softplus(z)``.

The residual scaling ``lora_alpha / r`` is applied to ``delta`` in the layer, not here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Gumbel-sigmoid sampling for differentiable binary decisions."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got: {tau}")

    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)  # Gumbel(0,1)
    y_soft = torch.sigmoid((logits + g) / tau)  # (0,1)

    if not hard:
        return y_soft.to(logits.dtype)

    y_hard = (y_soft > threshold).to(logits.dtype)
    return y_hard - y_soft.detach() + y_soft  # STE


def compute_gamma(
    z: torch.Tensor,
    *,
    gate: str,
    training: bool,
    gamma_threshold: float = 0.5,
    tau: float = 1.0,
) -> torch.Tensor:
    """Per-token gate value gamma from gate logits ``z``.

    hard: train -> Gumbel-STE binary sample; eval -> deterministic ``1[sigmoid(z) > threshold]``.
    soft: ``softplus(z)``.
    """
    if gate == "soft":
        return F.softplus(z)
    if gate != "hard":
        raise ValueError(f"gate must be 'hard' or 'soft', got: {gate}")

    if not training:
        return (torch.sigmoid(z) > gamma_threshold).to(z.dtype)
    return gumbel_sigmoid(z, tau=tau, hard=True, threshold=gamma_threshold)
