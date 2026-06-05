"""Runtime extension of ``peft``'s closed ``PeftType`` enum with a ``GADRA`` member.

``peft.utils.register_peft_method`` (peft 0.16.0) refuses to register a method whose
upper-cased name is not already a ``PeftType`` member (``peft/utils/peft_types.py:140``).
``PeftType`` is a ``str``-mixin ``enum.Enum`` and therefore closed at definition time, so we
splice a new member in at import. This is the documented enum-extension technique adapted for a
``str`` mix-in: build the member through the mix-in's ``__new__`` and register it in the enum's
internal maps, bypassing ``EnumType.__setattr__`` (which forbids assigning member names) via
``type.__setattr__``.

The shim is idempotent and side-effect-isolated to adding one member. When GaDRA is eventually
upstreamed into ``peft`` proper, ``PeftType.GADRA`` would be a normal enum entry and this module
would be deleted.
"""

from __future__ import annotations

import peft.utils.peft_types as _peft_types

GADRA_MEMBER_NAME = "GADRA"
GADRA_MEMBER_VALUE = "GADRA"


def ensure_gadra_peft_type():
    """Idempotently add ``GADRA`` to ``peft``'s ``PeftType`` enum and return the member."""
    peft_type_cls = _peft_types.PeftType

    existing = peft_type_cls.__members__.get(GADRA_MEMBER_NAME)
    if existing is not None:
        return existing

    # Build the member through the str mix-in's constructor (Enum.__new__ is overridden and
    # cannot be called directly to mint members post-definition).
    member = str.__new__(peft_type_cls, GADRA_MEMBER_VALUE)
    member._name_ = GADRA_MEMBER_NAME
    member._value_ = GADRA_MEMBER_VALUE

    # Register in the enum internals. type.__setattr__ bypasses EnumType.__setattr__, which would
    # otherwise raise "cannot reassign member" once the name is in _member_map_.
    type.__setattr__(peft_type_cls, GADRA_MEMBER_NAME, member)
    peft_type_cls._member_map_[GADRA_MEMBER_NAME] = member
    peft_type_cls._value2member_map_[GADRA_MEMBER_VALUE] = member
    if GADRA_MEMBER_NAME not in peft_type_cls._member_names_:
        peft_type_cls._member_names_.append(GADRA_MEMBER_NAME)

    return member
