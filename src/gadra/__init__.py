"""GaDRA — Gated Dual-conditioned Residual Adapter, a HuggingFace ``peft``-native custom tuner.

Importing this package registers the ``gadra`` method with peft, usable through the standard peft
API (``get_peft_model`` / ``PeftModel`` with :class:`GaDRAConfig`). Non-mergeable by design.
"""

from __future__ import annotations

import importlib.util

from ._enum_shim import ensure_gadra_peft_type

__version__ = "1.0.0"

ensure_gadra_peft_type()
from .config import GaDRAConfig  # noqa: E402  (must follow the enum shim)

__all__ = ["GaDRAConfig", "__version__"]

# register the full method once the tuner model is importable
if importlib.util.find_spec("gadra.model") is not None:
    from ._register import register_gadra

    register_gadra()
    from .model import GaDRAModel  # noqa: E402

    __all__ += ["GaDRAModel"]
