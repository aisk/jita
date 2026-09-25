"""x64 memory operands and size prefixes.

    qword[rbx + rcx*8 + 16]     size-tagged memory operand
    ptr[rbx]                    size taken from the other operand
    qword[rip + label]          rip-relative to a Label (REL32 patch)
    qword[rip + "name"]         same, naming a label of the assembler
    qword[rip + extern]         the 8 byte slot holding an Extern's address
    qword[0x1000]               absolute disp32 (SIB form, no base)
    qword[0x100000000]          absolute 64 bit address, only for mov64

The value of a size prefix is an instance of the matching subclass of
`MemExpr` (`qword[...]` is a `Mem64`, `ptr[...]` and `rbx + 8` are
`MemAny`), so type checkers can match memory operands by width.
"""

from dataclasses import dataclass, replace
from typing import Self

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Operand
from .regs import Reg, rip

__all__ = [
    "MemExpr",
    "MemAny",
    "Mem8",
    "Mem16",
    "Mem32",
    "Mem64",
    "Mem80",
    "Mem128",
    "Mem256",
    "Mem",
    "SizePrefix",
    "byte",
    "word",
    "dword",
    "qword",
    "tword",
    "oword",
    "yword",
    "ptr",
]

_SIZE_NAMES = {1: "byte", 2: "word", 4: "dword", 8: "qword", 10: "tbyte", 16: "xmmword", 32: "ymmword"}


@dataclass(frozen=True, slots=True, repr=False)
class MemExpr(Operand):
    """`[base + index*scale + disp]`, or `[rip + label + disp]`.

    A `label` that is an Extern refers to the extern's pointer slot
    (`Assembler.extern_slot`), so `call(qword[rip + ext])` is an indirect
    call through the slot. Such an operand cannot have a displacement.

    `size` is the access width in bytes, None when untyped (`ptr[...]`).
    Instances are normalized and validated on construction.

    Operands built with the size prefixes and register arithmetic are
    instances of the subclasses `Mem8` .. `Mem256` and `MemAny`. Equality
    ignores the subclass: it compares the fields.
    """

    base: Reg | None = None
    index: Reg | None = None
    scale: int = 1
    disp: int = 0
    label: Label | Extern | str | None = None  # str names a label of the assembler
    size: int | None = None

    def __post_init__(self) -> None:
        base, index, scale = self.base, self.index, self.scale
        if scale not in (1, 2, 4, 8):
            raise EncodeError(f"scale must be 1, 2, 4 or 8, got {scale!r}")
        # rsp can only be a base; with scale 1 swapping base and index is exact.
        if index is not None and index.kind == "gp" and index.code == 4 and scale == 1:
            if base is None:
                base, index = index, None
            elif base.kind == "gp" and base.code != 4:
                base, index = index, base
            object.__setattr__(self, "base", base)
            object.__setattr__(self, "index", index)
        if base is not None:
            if base.kind not in ("gp", "rip") or (base.kind == "gp" and base.size not in (4, 8)):
                raise EncodeError(f"{base} cannot be a memory base")
        if index is not None:
            if index.kind == "gp":
                if index.size not in (4, 8):
                    raise EncodeError(f"{index} cannot be a memory index")
                if index.code == 4:
                    raise EncodeError(f"{index} cannot be a memory index")
                if base is not None and base.kind == "gp" and base.size != index.size:
                    raise EncodeError(f"mixed address sizes: {base} and {index}")
            elif index.kind in ("xmm", "ymm"):
                raise EncodeError(f"VSIB addressing ({index} as index) is not supported")
            else:
                raise EncodeError(f"{index} cannot be a memory index")
        if base is rip and index is not None:
            raise EncodeError("rip-relative operands cannot have an index")
        if self.label is not None and base is not rip:
            raise EncodeError("labels are only allowed as rip-relative operands: [rip + label]")
        if isinstance(self.label, Extern) and self.disp:
            # The operand addresses the extern's pointer slot, so an offset
            # would point past the slot rather than into the extern.
            raise EncodeError(
                f"[rip + {self.label.name}] addresses the extern's 8 byte pointer slot, "
                "it cannot have a displacement"
            )
        disp = self.disp
        if not isinstance(disp, int) or isinstance(disp, bool):
            raise EncodeError(f"displacement {disp!r} is not an int")
        if base is None and index is None and self.label is None:
            # Pure absolute address: 64 bits are allowed, but only mov64
            # (moffs64) can encode more than a sign-extended disp32.
            if not -(1 << 63) <= disp < (1 << 64):
                raise EncodeError(f"absolute address {disp:#x} does not fit in 64 bits")
        elif not -(1 << 31) <= disp < (1 << 31):
            raise EncodeError(f"displacement {disp!r} does not fit in int32")

    def _key(self) -> tuple[object, ...]:
        return (self.base, self.index, self.scale, self.disp, self.label, self.size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemExpr):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def _add_reg(self, reg: Reg, scale: int = 1) -> Self:
        if scale == 1 and self.base is None:
            return replace(self, base=reg)
        if self.index is None:
            return replace(self, index=reg, scale=scale)
        raise EncodeError(f"too many registers in memory operand: {self} + {reg}")

    def __add__(self, other: int | Reg | MemExpr | Label | Extern | str) -> Self:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return replace(self, disp=self.disp + other)
        if isinstance(other, (Label, Extern, str)):
            if self.label is not None:
                raise EncodeError("memory operand can reference only one label")
            return replace(self, label=other)
        if isinstance(other, Reg):
            return self._add_reg(other)
        if isinstance(other, MemExpr):
            if self.size is not None or other.size is not None:
                raise EncodeError("cannot add sized memory operands")
            m = self
            if other.base is not None:
                m = m._add_reg(other.base)
            if other.index is not None:
                m = m._add_reg(other.index, other.scale)
            if other.label is not None:
                m = m + other.label
            return m + other.disp
        return NotImplemented

    def __radd__(self, other: int | Reg | MemExpr | Label | Extern | str) -> Self:
        return self.__add__(other)

    def __sub__(self, other: int) -> Self:
        if isinstance(other, int) and not isinstance(other, bool):
            return replace(self, disp=self.disp - other)
        return NotImplemented

    def __str__(self) -> str:
        parts: list[str] = []
        if self.base is not None:
            parts.append(self.base.name)
        if self.index is not None:
            parts.append(f"{self.index.name}*{self.scale}")
        if self.label is not None:
            parts.append(str(self.label))
        text = "+".join(parts)
        if self.disp or not text:
            text += f"{self.disp:+d}" if text else str(self.disp)
        prefix = "" if self.size is None else f"{_SIZE_NAMES.get(self.size, self.size)} ptr "
        return f"{prefix}[{text}]"

    __repr__ = __str__


class MemAny(MemExpr):
    """Memory operand without a size (`ptr[...]`, `rbx + 8`). The size
    comes from the other operands."""

    __slots__ = ()


class Mem8(MemExpr):
    """Byte memory operand (`byte[...]`)."""

    __slots__ = ()


class Mem16(MemExpr):
    """Word memory operand (`word[...]`)."""

    __slots__ = ()


class Mem32(MemExpr):
    """Dword memory operand (`dword[...]`)."""

    __slots__ = ()


class Mem64(MemExpr):
    """Qword memory operand (`qword[...]`)."""

    __slots__ = ()


class Mem80(MemExpr):
    """Ten byte memory operand (`tword[...]`, x87)."""

    __slots__ = ()


class Mem128(MemExpr):
    """16 byte memory operand (`oword[...]`)."""

    __slots__ = ()


class Mem256(MemExpr):
    """32 byte memory operand (`yword[...]`)."""

    __slots__ = ()


# Any memory operand built with a size prefix or register arithmetic, for
# annotating helpers.
type Mem = Mem8 | Mem16 | Mem32 | Mem64 | Mem80 | Mem128 | Mem256 | MemAny

_MEM_BY_SIZE: dict[int | None, type[MemExpr]] = {
    None: MemAny,
    1: Mem8,
    2: Mem16,
    4: Mem32,
    8: Mem64,
    10: Mem80,
    16: Mem128,
    32: Mem256,
}


def _resized[M: MemExpr](m: MemExpr, cls: type[M], size: int | None) -> M:
    return cls(m.base, m.index, m.scale, m.disp, m.label, size)


def sized(m: MemExpr, size: int | None) -> MemExpr:
    """`m` with access width `size` (bytes, None for unsized), as an
    instance of the matching `MemExpr` subclass."""
    return _resized(m, _MEM_BY_SIZE.get(size, MemExpr), size)


class SizePrefix[M: MemExpr]:
    """`qword[...]` etc. Indexing produces a sized memory operand of class M."""

    __slots__ = ("name", "size", "cls")

    def __init__(self, name: str, size: int | None, cls: type[M]):
        self.name, self.size, self.cls = name, size, cls

    def __getitem__(self, x: Reg | MemExpr | int) -> M:
        if isinstance(x, MemExpr):
            m = x
        elif isinstance(x, Reg):
            m = MemExpr(base=x)
        elif isinstance(x, int) and not isinstance(x, bool):
            m = MemExpr(disp=x)
        elif isinstance(x, (Label, Extern, str)):
            raise EncodeError(f"{self.name}[label] is not addressable on x64, use {self.name}[rip + label]")
        else:
            raise TypeError(f"{self.name}[...] expects a register, int or memory expression, got {x!r}")
        return _resized(m, self.cls, self.size)

    def __repr__(self) -> str:
        return self.name


byte = SizePrefix("byte", 1, Mem8)
word = SizePrefix("word", 2, Mem16)
dword = SizePrefix("dword", 4, Mem32)
qword = SizePrefix("qword", 8, Mem64)
tword = SizePrefix("tword", 10, Mem80)
oword = SizePrefix("oword", 16, Mem128)
yword = SizePrefix("yword", 32, Mem256)
ptr = SizePrefix("ptr", None, MemAny)
