"""loongarch64 memory operands, made by `typed()` or built directly.

LoongArch assembly has no memory operand syntax: a load names its base
register and offset as two operands, `ld.w $a0, $a1, 8`, and an indexed
load its base and index registers, `ldx.w $a0, $a1, $a2`. jita keeps that
spelling for plain code. `typed()` views memory as a ctypes type, and its
fields and elements are `MemExpr`s that stand for those two operands:
`ld.w(a0, p.x)` encodes `ld.w $a0, $a1, 8` for a field `x` at offset 8 of
`typed(a1, T)`. A `MemExpr` is one of

    [$a1, 8]        base and offset: ld.*, st.*, fld.*, fst.*, preld,
                    ldptr.*, stptr.*, ll.*, sc.*
    [$a1, $a2]      base and index register: ldx.*, stx.*, fldx.*,
                    fstx.*, preldx
    [$a1]           base alone, also the bare base of am*, ldgt.*,
                    ldle.*, stgt.*, stle.* and their fld/fst forms

A `MemExpr` built directly, `MemExpr(a1, 8)` or `MemExpr(a1, index=a2)`,
has no size and encodes exactly as the operand pair it stands for. The
operands of `typed()` carry the field's size, and the encoder rejects an
instruction whose access width differs. `a1 + 8` builds an address for
`typed(a1 + 8, T)`.
"""

from dataclasses import dataclass
from typing import TypeGuard

from ..core.errors import EncodeError
from ..core.operand import Operand
from .regs import R, Reg


def _is_int(x: object) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


class Addr:
    """An address under construction: `a1 + 8`, the base of `typed()`."""

    __slots__ = ("base", "disp")

    def __init__(self, base: Reg, disp: int = 0):
        self.base, self.disp = base, disp

    def __add__(self, other: int) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.disp + other)
        if isinstance(other, (Reg, Addr)):
            raise EncodeError(
                f"a loongarch64 address is a base register plus an offset, not {self} + {other}; "
                "index an array view with the register instead"
            )
        return NotImplemented

    def __radd__(self, other: int) -> Addr:
        return self.__add__(other)

    def __sub__(self, other: int) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.disp - other)
        return NotImplemented

    def __str__(self) -> str:
        if self.disp < 0:
            return f"{self.base} - {-self.disp}"
        return f"{self.base} + {self.disp}" if self.disp else str(self.base)

    __repr__ = __str__


@dataclass(frozen=True, slots=True, repr=False)
class MemExpr(Operand):
    """A base register with an offset or an index register. Validated on
    construction; the offset range is checked by the encoder.

    `size` is None for the unsized `addr` of a view and for a `MemExpr`
    built directly. `typed()` sets it to
    the field size (1, 2, 4 or 8), and the encoder then requires the
    instruction to access that many bytes."""

    base: R
    disp: int = 0
    index: R | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, R):
            raise EncodeError(f"{self.base} cannot be a memory base (use an integer register)")
        if not _is_int(self.disp):
            raise EncodeError(f"offset {self.disp!r} is not an int")
        if self.index is not None:
            if not isinstance(self.index, R):
                raise EncodeError(f"{self.index} cannot be a memory index (use an integer register)")
            if self.disp:
                raise EncodeError("a loongarch64 address cannot have both an index register and an offset")
        if self.size is not None and not (_is_int(self.size) and self.size in (1, 2, 4, 8)):
            raise EncodeError(f"memory operand size {self.size!r} is not 1, 2, 4 or 8")

    def __str__(self) -> str:
        if self.index is not None:
            return f"[{self.base}, {self.index}]"
        return f"[{self.base}, {self.disp}]" if self.disp else f"[{self.base}]"

    __repr__ = __str__


__all__ = ["MemExpr"]
