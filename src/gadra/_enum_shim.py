"""Runtime extension of ``peft``'s closed ``PeftType`` enum with a ``GADRA`` member.

``register_peft_method`` (peft 0.19.1) requires the method's upper-cased name to be an existing
``PeftType`` member, so this splices ``GADRA`` into the enum at import. Idempotent.
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

    # build the member through the str mix-in's constructor
    member = str.__new__(peft_type_cls, GADRA_MEMBER_VALUE)
    member._name_ = GADRA_MEMBER_NAME
    member._value_ = GADRA_MEMBER_VALUE

    # register in the enum internals (type.__setattr__ bypasses EnumType's member-assignment guard)
    type.__setattr__(peft_type_cls, GADRA_MEMBER_NAME, member)
    peft_type_cls._member_map_[GADRA_MEMBER_NAME] = member
    peft_type_cls._value2member_map_[GADRA_MEMBER_VALUE] = member
    if GADRA_MEMBER_NAME not in peft_type_cls._member_names_:
        peft_type_cls._member_names_.append(GADRA_MEMBER_NAME)

    return member
