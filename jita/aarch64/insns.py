"""Module-level aarch64 mnemonic functions, generated from the template table.

Every mnemonic in `jita.aarch64.table.MAP_OP` becomes a function
`name(*ops, asm=None)` that encodes into `asm`, or into the current
Assembler when `asm` is None. `and` and `str` are spelled `and_` and
`str_` (a Python keyword and a builtin); `a.str(...)` also works as a
method.

Conditional branches exist both as DynASM spells them (`beq`, `bne`,
`bhs`, ...) and as attributes of `b`: `b.eq(lbl)`, `b.ne("loop")`.

`movz`, `movn` and `movk` take the halfword shift as a keyword:
`movk(x0, 0x1234, lsl=16)` is `movk x0, #0x1234, lsl #16`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.assembler import current
from .encoder import MNEMONIC_ARGC, encode
from .regs import Mod
from .table import MAP_COND

# Table name -> Python name for mnemonics that are keywords or builtins.
PY_NAMES = {"and": "and_", "str": "str_"}

_WIDE = frozenset(["movz", "movn", "movk"])


def _make(mnemonic: str, pyname: str, shown: str | None = None) -> Callable[..., None]:
    shown = shown or mnemonic
    if mnemonic in _WIDE:

        def insn(*ops: Any, lsl: int | None = None, asm: Any = None) -> None:
            if lsl is not None:
                ops = (*ops, Mod("lsl", lsl))
            encode(asm or current(), mnemonic, ops, shown)

    else:

        def insn(*ops: Any, asm: Any = None) -> None:
            encode(asm or current(), mnemonic, ops, shown)

    insn.__name__ = insn.__qualname__ = pyname
    counts = MNEMONIC_ARGC.get(mnemonic, ())
    desc = "any number of" if "*" in counts else " or ".join(map(str, counts))
    insn.__doc__ = f"Emit `{shown}` ({desc} operands)."
    return insn


# Python name -> function. The Assembler delegates `a.<name>(...)` here.
INSNS: dict[str, Callable[..., None]] = {}

for _mn in sorted(MNEMONIC_ARGC):
    _py = PY_NAMES.get(_mn, _mn)
    INSNS[_py] = globals()[_py] = _make(_mn, _py)
del _mn, _py

# b.eq, b.ne, ... as attributes of b (same encoding as beq, bne, ...).
for _cond in MAP_COND:
    setattr(INSNS["b"], _cond, _make("b" + _cond, _cond, "b." + _cond))
del _cond

# `a.str(...)` is legal syntax, so allow the builtin-shadowing name as a method.
INSNS["str"] = INSNS["str_"]

__all__ = sorted(k for k in INSNS if k != "str")
