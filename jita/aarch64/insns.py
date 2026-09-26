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

Type checkers read `insns.pyi` next to this module, which
`tools/gen_stubs.py` generates with one overload per accepted combination
of operand classes.
"""

# __all__ is computed; the stub next to this module carries it literally.
# pyright: reportUnsupportedDunderAll=false

from collections.abc import Callable
from typing import Any

from ..core.assembler import Assembler, current
from .encoder import MNEMONIC_ARGC, encode
from .regs import Mod
from .table import MAP_COND

# Table name -> Python name for mnemonics that are keywords or builtins.
PY_NAMES = {"and": "and_", "str": "str_"}

_WIDE = frozenset(["movz", "movn", "movk"])


def _make(mnemonic: str, pyname: str, shown: str | None = None) -> Callable[..., None]:
    shown = shown or mnemonic
    counts = MNEMONIC_ARGC.get(mnemonic, ())
    desc = "any number of" if "*" in counts else " or ".join(map(str, counts))
    doc = f"Emit `{shown}` ({desc} operands)."
    if mnemonic in _WIDE:

        def wide(*ops: Any, lsl: int | None = None, asm: Any = None) -> None:
            if lsl is not None:
                ops = (*ops, Mod("lsl", lsl))
            encode(asm or current(), mnemonic, ops, shown)

        wide.__name__ = wide.__qualname__ = pyname
        wide.__doc__ = doc
        return wide

    def insn(*ops: Any, asm: Any = None) -> None:
        encode(asm or current(), mnemonic, ops, shown)

    insn.__name__ = insn.__qualname__ = pyname
    insn.__doc__ = doc
    return insn


# Python name -> function. The Assembler delegates `a.<name>(...)` here.
INSNS: dict[str, Callable[..., None]] = {}


def _define() -> None:
    names = globals()
    for mn in sorted(MNEMONIC_ARGC):
        py = PY_NAMES.get(mn, mn)
        # beq and b.eq are one instruction; listings always show b.eq.
        shown = "b." + mn[1:] if mn[0] == "b" and mn[1:] in MAP_COND else None
        INSNS[py] = names[py] = _make(mn, py, shown)
    # b.eq, b.ne, ... as attributes of b (same encoding as beq, bne, ...).
    for cond in MAP_COND:
        setattr(INSNS["b"], cond, _make("b" + cond, cond, "b." + cond))


_define()

# `a.str(...)` is legal syntax, so allow the builtin-shadowing name as a method.
INSNS["str"] = INSNS["str_"]

__all__ = sorted(k for k in INSNS if k != "str")


class Aarch64Assembler(Assembler):
    """An Assembler for aarch64. `Assembler("aarch64")` returns one; its
    mnemonic methods are typed in the stub."""

    _default_arch = "aarch64"
