"""Module-level riscv64 mnemonic functions, generated from the template table.

Every mnemonic in `jita.riscv64.table.MAP_OP` becomes a function
`name(*ops, asm=None)` that encodes into `asm`, or into the current
Assembler when `asm` is None. `and`, `or` and `not` are spelled `and_`,
`or_` and `not_` (Python keywords), `min` and `max` are `min_` and `max_`
(builtins); `a.min(...)` and `a.max(...)` also work as methods.

Mnemonics with dots are attributes: `fadd.d(fa0, fa1, fa2)`,
`fcvt.w.d(a0, fa0, rm="rtz")`, `lr.w.aq(a0, mem[a1])`, `fence.i()`.
`fadd`, `fcvt` and the other prefixes that are not instructions
themselves are namespaces; `fence`, `fmv.d` and `lr.w` are both an
instruction and a namespace.

Floating point instructions that round take the rounding mode as the
keyword `rm` ("rne", "rtz", "rdn", "rup", "rmm" or "dyn", the default).

Type checkers read `insns.pyi` next to this module, which
`tools/gen_stubs.py` generates with one overload per accepted combination
of operand classes.
"""

# __all__ is computed; the stub next to this module carries it literally.
# pyright: reportUnsupportedDunderAll=false

from collections.abc import Callable
from typing import Any

from ..core.assembler import Assembler, current
from ..core.errors import EncodeError
from .encoder import MNEMONIC_ARGC, RM_MNEMONICS, encode

# Table name -> Python name for mnemonics that are keywords or builtins.
PY_NAMES = {"and": "and_", "or": "or_", "not": "not_", "min": "min_", "max": "max_"}


class Group:
    """The mnemonics that share a dotted prefix, `fadd` of `fadd.s` and
    `fadd.d`, as attributes. Calling it is an error: the prefix is not an
    instruction."""

    def __init__(self, name: str):
        self.__name__ = self.__qualname__ = name
        self.__doc__ = f"Namespace of the `{name}.*` mnemonics."

    def __call__(self, *ops: Any, **kw: Any) -> None:
        subs = sorted(k for k, v in vars(self).items() if not k.startswith("_") and callable(v))
        alts = ", ".join(f"{self.__name__}.{s}" for s in subs)
        raise EncodeError(f"{self.__name__} is not an instruction, use {alts}")

    def __repr__(self) -> str:
        return f"<riscv64 mnemonics {self.__name__}.*>"


def _make(mnemonic: str, pyname: str) -> Callable[..., None]:
    counts = MNEMONIC_ARGC[mnemonic]
    doc = f"Emit `{mnemonic}` ({' or '.join(map(str, counts))} operands)."
    if mnemonic in RM_MNEMONICS:

        def rounding(*ops: Any, rm: str | None = None, asm: Any = None) -> None:
            encode(asm or current(), mnemonic, ops, rm)

        fn = rounding
        doc = doc[:-2] + ", rounding mode `rm=`)."
    else:

        def insn(*ops: Any, asm: Any = None) -> None:
            encode(asm or current(), mnemonic, ops)

        fn = insn
    fn.__name__ = pyname
    fn.__qualname__ = mnemonic if "." in mnemonic else pyname
    fn.__doc__ = doc
    setattr(fn, "__test__", False)
    return fn


# Python name -> function or Group. The Assembler delegates `a.<name>(...)`
# here; `a.fadd.d(...)` goes through the Group.
INSNS: dict[str, Callable[..., None]] = {}

# Mnemonic ("fadd.d") -> function, dotted names included.
MNEMONICS: dict[str, Callable[..., None]] = {}


def _define() -> None:
    names = globals()
    # Shorter names first: fmv.d is created before fmv.d.x hangs off it.
    for mn in sorted(MNEMONIC_ARGC, key=lambda m: (m.count("."), m)):
        parts = mn.split(".")
        fn = MNEMONICS[mn] = _make(mn, PY_NAMES.get(parts[-1], parts[-1]) if len(parts) > 1 else PY_NAMES.get(mn, mn))
        if len(parts) == 1:
            INSNS[fn.__name__] = names[fn.__name__] = fn
            continue
        parent: Any = INSNS.get(parts[0])
        if parent is None:
            parent = INSNS[parts[0]] = names[parts[0]] = Group(parts[0])
        for i, part in enumerate(parts[1:-1], 1):
            child = getattr(parent, part, None)
            if child is None:
                child = Group(".".join(parts[: i + 1]))
                setattr(parent, part, child)
            parent = child
        setattr(parent, parts[-1], fn)


_define()

# `a.min(...)` is legal syntax, so allow the builtin-shadowing names as methods.
INSNS["min"] = INSNS["min_"]
INSNS["max"] = INSNS["max_"]

__all__ = sorted(k for k in INSNS if k not in ("min", "max"))


class Riscv64Assembler(Assembler):
    """An Assembler for riscv64. `Assembler("riscv64")` returns one; its
    mnemonic methods are typed in the stub."""

    _default_arch = "riscv64"
