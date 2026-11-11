"""Factorized decoder (plan §6.5): maps a settled value to a token in the question's
context. It never mixes parts of an answer: one value, one token."""
from __future__ import annotations

from ..memory.stores import Context
from ..world.api import FALSE, TRUE


def decode(ctx: Context, value: int, boolean: bool) -> str:
    if boolean:
        return TRUE if value else FALSE
    return ctx.tokens[value]
