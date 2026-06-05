"""Gate functions for GaDRA (method-critical).

``gumbel_sigmoid`` is copied verbatim from the research repo's ``peft_losses.gumbel_sigmoid`` — it is
the Gumbel-sigmoid straight-through estimator that produces the binary hard gate during training.
``compute_gamma`` reproduces the released ``PeftBlock._gate_output_fn_forward`` decision for the two
shipped gate types:

* **hard** (``gamma_hard_masking="Gumbel"`` in the legacy config): training draws a Gumbel-STE
  binary sample; inference is the deterministic threshold ``1[sigmoid(z) > threshold]``.
* **soft** (legacy ``gamma_hard_masking=None``): continuous ``softplus(z)`` — NOTE this is the
  released code path that produced the paper's soft-gate numbers; the paper text describes a
  ``sigmoid`` in ``[0,1]``. We replicate the code-as-run softplus and document the discrepancy.

The residual scale ``alpha`` (the released ``output_scalar``) is applied to ``delta`` in the layer,
not here, so these functions return the raw per-token gate value.
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
    """Gumbel-Sigmoid sampling for differentiable binary decisions (verbatim from peft_losses)."""
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
    soft: ``softplus(z)`` (code-as-run; see module docstring).
    """
    if gate == "soft":
        return F.softplus(z)
    if gate != "hard":
        raise ValueError(f"gate must be 'hard' or 'soft', got: {gate}")

    if not training:
        return (torch.sigmoid(z) > gamma_threshold).to(z.dtype)
    return gumbel_sigmoid(z, tau=tau, hard=True, threshold=gamma_threshold)
