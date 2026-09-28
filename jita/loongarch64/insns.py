"""Module-level loongarch64 mnemonic functions, generated from the template table.

Every mnemonic in `jita.loongarch64.table.MAP_OP` becomes a function
`name(*ops, asm=None)` that encodes into `asm`, or into the current
Assembler when `asm` is None. `and`, `or` and `break` are spelled `and_`,
`or_` and `break_` (Python keywords).

Mnemonics with dots are attributes: `add.d(a0, a1, a2)`,
`fcmp.clt.d(fcc0, fa0, fa1)`, `amadd_db.w(a0, a1, a2)`,
`crc.w.b.w(a0, a1, a2)`. `add`, `ld`, `fcmp` and the other prefixes are
namespaces, not instructions. A part that starts with a digit gets a
leading underscore: `revb._2h`, `bitrev._4b`.

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
from .encoder import MNEMONIC_ARGC, encode

# Table name -> Python name for mnemonics that are keywords.
PY_NAMES = {"and": "and_", "or": "or_", "break": "break_"}


def py_part(part: str) -> str:
    """The Python spelling of one dotted part of a mnemonic: `2h` is
    `_2h`, the other parts are unchanged."""
    return "_" + part if part[:1].isdigit() else part


class Group:
    """The mnemonics that share a dotted prefix, `add` of `add.w` and
    `add.d`, as attributes. Calling it is an error: the prefix is not an
    instruction."""

    def __init__(self, name: str):
        self.__name__ = self.__qualname__ = name
        self.__doc__ = f"Namespace of the `{name}.*` mnemonics."

    def __call__(self, *ops: Any, **kw: Any) -> None:
        # The Python spelling, with the GNU one where they differ.
        subs = []
        for k, v in sorted(vars(self).items()):
            if k.startswith("__") or not callable(v):
                continue
            py = f"{self.__name__}.{k}"
            subs.append(py if py == v.__qualname__ else f"{py} ({v.__qualname__})")
        raise EncodeError(f"{self.__name__} is not an instruction, use {', '.join(subs)}")

    def __repr__(self) -> str:
        return f"<loongarch64 mnemonics {self.__name__}.*>"


def _make(mnemonic: str, pyname: str) -> Callable[..., None]:
    counts = MNEMONIC_ARGC[mnemonic]

    def insn(*ops: Any, asm: Any = None) -> None:
        encode(asm or current(), mnemonic, ops)

    insn.__name__ = pyname
    insn.__qualname__ = mnemonic if "." in mnemonic else pyname
    insn.__doc__ = f"Emit `{mnemonic}` ({' or '.join(map(str, counts))} operands)."
    setattr(insn, "__test__", False)
    return insn


# Python name -> function or Group. The Assembler delegates `a.<name>(...)`
# here; `a.add.d(...)` goes through the Group.
INSNS: dict[str, Callable[..., None]] = {}

# Mnemonic ("add.d") -> function, dotted names included.
MNEMONICS: dict[str, Callable[..., None]] = {}


def _define() -> None:
    names = globals()
    for mn in sorted(MNEMONIC_ARGC, key=lambda m: (m.count("."), m)):
        parts = mn.split(".")
        if len(parts) == 1:
            fn = MNEMONICS[mn] = _make(mn, PY_NAMES.get(mn, mn))
            INSNS[fn.__name__] = names[fn.__name__] = fn
            continue
        fn = MNEMONICS[mn] = _make(mn, py_part(parts[-1]))
        parent: Any = INSNS.get(parts[0])
        if parent is None:
            parent = INSNS[parts[0]] = names[parts[0]] = Group(parts[0])
        for i, part in enumerate(parts[1:-1], 1):
            child = getattr(parent, py_part(part), None)
            if child is None:
                child = Group(".".join(parts[: i + 1]))
                setattr(parent, py_part(part), child)
            parent = child
        setattr(parent, fn.__name__, fn)


_define()

__all__ = sorted(INSNS)


class Loongarch64Assembler(Assembler):
    """An Assembler for loongarch64. `Assembler("loongarch64")` returns
    one; its mnemonic methods are typed in the stub."""

    _default_arch = "loongarch64"
