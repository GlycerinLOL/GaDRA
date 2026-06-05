"""GaDRA — Gated Dual-conditioned Residual Adapter, as a HuggingFace ``peft``-native custom tuner.

Importing this package makes :class:`GaDRAConfig` available and adds ``PeftType.GADRA`` to peft.
Once the tuner model (:mod:`gadra.model`, added in P2) is present, the ``gadra`` method is fully
registered and usable through the standard peft API::

    from gadra import GaDRAConfig
    from peft import get_peft_model
    model = get_peft_model(base_model, GaDRAConfig())

GaDRA is non-mergeable by design (the residual gate is per-token and input-dependent).
"""

from __future__ import annotations

import importlib.util

from ._enum_shim import ensure_gadra_peft_type

__version__ = "0.0.0.dev0"

# Always available: the config and the PeftType.GADRA enum member.
ensure_gadra_peft_type()
from .config import GaDRAConfig  # noqa: E402  (must follow the enum shim)

__all__ = ["GaDRAConfig", "__version__"]

# Full method registration needs GaDRAModel (gadra.model, P2). Activate it automatically as soon as
# that module exists, without forcing the import here when it does not (keeps P0/P1 self-contained).
if importlib.util.find_spec("gadra.model") is not None:
    from ._register import register_gadra

    register_gadra()
    from .model import GaDRAModel  # noqa: E402

    __all__ += ["GaDRAModel"]
