"""riscv64 memory operands.

    mem[a0]                     0(a0)
    mem[sp + 8], mem[a0 - 16]   8(sp), -16(a0)

RISC-V has one addressing mode, a base register plus a signed 12 bit
offset; the encoder checks the range. There are no size prefixes: the
access width comes from the instruction (`lb`, `lhu`, `ld`, `flw`,
`fsd`). The operands that `typed()` makes for ctypes fields carry the
field's size, and the encoder rejects an instruction whose access width
differs. The atomic instructions take `mem[a0]` with no offset, written
`(a0)` in GNU syntax. Load a label with `ld(a0, lbl)` or its address with
`lla(a0, lbl)` instead of a memory operand.
"""

from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from ..core.errors import EncodeError
from ..core.operand import Operand
from .regs import Reg, X


def _is_int(x: object) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


class Addr:
    """An address expression under construction: `sp + 8`.

    Not an operand by itself; wrap it with `mem[...]`.
    """

    __slots__ = ("base", "disp")

    def __init__(self, base: Reg, disp: int = 0):
        self.base, self.disp = base, disp

    def __add__(self, other: int) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.disp + other)
        if isinstance(other, (Reg, Addr)):
            raise EncodeError(f"riscv64 addresses have no index register: {self} + {other}")
        return NotImplemented

    def __radd__(self, other: int) -> Addr:
        return self.__add__(other)

    def __sub__(self, other: int) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.disp - other)
        return NotImplemented

    def __str__(self) -> str:
        return f"{self.base} + {self.disp}" if self.disp else str(self.base)

    __repr__ = __str__


@dataclass(frozen=True, slots=True, repr=False)
class MemExpr(Operand):
    """`disp(base)`: an integer base register plus an offset. Validated on
    construction; the offset range (signed 12 bit, 0 for atomics) is
    checked by the encoder.

    `size` is None for `mem[...]`. `typed()` sets it to the field size
    (1, 2, 4 or 8), and the encoder then requires the instruction to access
    that many bytes. It is left out of the text."""

    base: X
    disp: int = 0
    size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, X):
            raise EncodeError(f"{self.base} cannot be a memory base (use an integer register)")
        if not _is_int(self.disp):
            raise EncodeError(f"offset {self.disp!r} is not an int")
        if self.size is not None and not (_is_int(self.size) and self.size in (1, 2, 4, 8)):
            raise EncodeError(f"memory operand size {self.size!r} is not 1, 2, 4 or 8")

    def __str__(self) -> str:
        return f"{self.disp}({self.base})"

    __repr__ = __str__


class _Mem:
    """`mem[base]`, `mem[base + offset]`."""

    __slots__ = ()

    def __getitem__(self, key: X | Addr | MemExpr) -> MemExpr:
        x: Any = key
        if isinstance(x, Reg):
            return MemExpr(cast(X, x))  # validated by MemExpr
        if isinstance(x, Addr):
            return MemExpr(cast(X, x.base), x.disp)
        if isinstance(x, MemExpr):
            return x
        if isinstance(x, (Operand, str, tuple)) or _is_int(x):
            raise EncodeError(f"mem[...] expects a base register or base + offset, got {x!r}")
        raise TypeError(f"mem[...] expects a register or address expression, got {x!r}")

    def __repr__(self) -> str:
        return "mem"


mem = _Mem()

__all__ = ["MemExpr", "mem"]
