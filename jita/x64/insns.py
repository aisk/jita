"""Module-level x64 mnemonic functions, generated from the template table.

Every mnemonic in `jita.x64.table.MAP_OP` becomes a function
`name(*ops, asm=None)` that encodes into `asm`, or into the current
Assembler when `asm` is None. Mnemonics that clash with Python keywords or
builtins get a trailing underscore: `and_ or_ not_ int_`.

`jmp` and every `j<cc>` also carry a `.short` attribute that emits the rel8
form: `jmp.short(lbl)`, `jz.short(lbl)`.

`mov64` (alias `movabs`) is DynASM's x64-only 64 bit immediate / absolute
address move. Plain `mov r64, imm` switches to it automatically when the
immediate does not fit in a sign-extended int32 or is a Label/Extern.

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
from .table import MAP_CC

# Table name -> Python name for mnemonics that are keywords or builtins.
PY_NAMES = {"and": "and_", "or": "or_", "not": "not_", "int": "int_"}

# Mnemonics with a rel8 form (DynASM shrinks these in dasm_link).
SHORT_BRANCHES = frozenset(["jmp", *("j" + cc for cc in MAP_CC)])


def _make(mnemonic: str, pyname: str) -> Callable[..., None]:
    def insn(*ops: Any, asm: Any = None) -> None:
        encode(asm or current(), mnemonic, ops)

    insn.__name__ = insn.__qualname__ = pyname
    counts = MNEMONIC_ARGC.get(mnemonic, (2,))
    insn.__doc__ = f"Emit `{mnemonic}` ({' or '.join(map(str, counts))} operands)."

    if mnemonic in SHORT_BRANCHES:

        def short(*ops: Any, asm: Any = None) -> None:
            encode(asm or current(), mnemonic, ops, short=True)

        short.__name__ = "short"
        short.__qualname__ = f"{pyname}.short"
        short.__doc__ = f"Emit `{mnemonic}` with a rel8 displacement."
        setattr(insn, "short", short)
    return insn


# Python name -> function. The Assembler delegates `a.<name>(...)` here.
INSNS: dict[str, Callable[..., None]] = {}


def _define() -> None:
    names = globals()
    for mn in sorted([*MNEMONIC_ARGC, "mov64", "movabs"]):
        py = PY_NAMES.get(mn, mn)
        INSNS[py] = names[py] = _make(mn, py)


_define()

# `a.int(3)` is legal syntax, so allow the builtin-shadowing name as a method.
INSNS["int"] = INSNS["int_"]

__all__ = sorted(k for k in INSNS if k != "int")


class X64Assembler(Assembler):
    """An Assembler for x64. `Assembler("x64")` returns one; its mnemonic
    methods are typed in the stub."""

    _default_arch = "x64"


def nop_fill(n: int) -> bytes:
    """Exactly `n` bytes of NOP padding (see `jita.x64.ARCH.nop_fill`)."""
    from . import ARCH

    return ARCH.nop_fill(n)
