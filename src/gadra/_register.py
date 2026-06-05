"""Register GaDRA as a first-class ``peft`` method (Decision A).

After :func:`register_gadra` runs, GaDRA dispatches through the standard peft API
(``get_peft_model``, ``PeftModel.from_pretrained``, ``save_pretrained``) on ``config.peft_type``,
exactly like LoRA. Gate parameters are auto-saved via the ``gadra_`` prefix.

Registration is split out (not done at ``config`` import) because it requires ``GaDRAModel``
(the tuner, added in P2). :mod:`gadra.__init__` calls :func:`register_gadra` only once ``model.py``
is importable, so importing :class:`gadra.config.GaDRAConfig` alone never forces the model.
"""

from __future__ import annotations

import warnings

import peft

from ._enum_shim import ensure_gadra_peft_type

_EXPECTED_PEFT_VERSION = "0.16.0"
_registered = False


def register_gadra() -> None:
    """Idempotently register the ``gadra`` method with the installed ``peft`` library."""
    global _registered
    if _registered:
        return

    installed = getattr(peft, "__version__", "unknown")
    if installed != _EXPECTED_PEFT_VERSION:
        warnings.warn(
            f"gadra was validated against peft=={_EXPECTED_PEFT_VERSION}; found peft=={installed}. "
            "The PeftType enum shim and registration may behave differently on other versions.",
            stacklevel=2,
        )

    ensure_gadra_peft_type()

    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.utils import register_peft_method
    from peft.utils.peft_types import PeftType

    # Idempotent across re-imports: register_peft_method raises KeyError if already present.
    if PeftType.GADRA in PEFT_TYPE_TO_CONFIG_MAPPING:
        _registered = True
        return

    from .config import GaDRAConfig
    from .model import GaDRAModel

    register_peft_method(
        name="gadra",
        config_cls=GaDRAConfig,
        model_cls=GaDRAModel,
        prefix="gadra_",
    )
    _registered = True
