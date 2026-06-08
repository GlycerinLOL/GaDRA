"""Register GaDRA as a first-class ``peft`` method.

After :func:`register_gadra` runs, GaDRA dispatches through the standard peft API
(``get_peft_model``, ``PeftModel.from_pretrained``, ``save_pretrained``) on ``config.peft_type``.
Gate parameters are auto-saved via the ``gadra_`` prefix.
"""

from __future__ import annotations

import warnings

import peft

from ._enum_shim import ensure_gadra_peft_type

_EXPECTED_PEFT_VERSION = "0.16.0"


def register_gadra() -> None:
    """Idempotently register the ``gadra`` method with the installed ``peft`` library."""
    ensure_gadra_peft_type()

    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.utils import register_peft_method
    from peft.utils.peft_types import PeftType

    if PeftType.GADRA in PEFT_TYPE_TO_CONFIG_MAPPING:
        return  # already registered

    installed = getattr(peft, "__version__", "unknown")
    if installed != _EXPECTED_PEFT_VERSION:
        warnings.warn(
            f"gadra was validated against peft=={_EXPECTED_PEFT_VERSION}; found peft=={installed}. "
            "The PeftType enum shim and registration may behave differently on other versions.",
            stacklevel=2,
        )

    from .config import GaDRAConfig
    from .model import GaDRAModel

    register_peft_method(
        name="gadra",
        config_cls=GaDRAConfig,
        model_cls=GaDRAModel,
        prefix="gadra_",
    )
