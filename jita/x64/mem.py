"""x64 memory operands and size prefixes.

    qword[rbx + rcx*8 + 16]     size-tagged memory operand
    ptr[rbx]                    size taken from the other operand
    qword[rip + label]          rip-relative to a Label (REL32 patch)
    qword[rip + "name"]         same, naming a label of the assembler
    qword[0x1000]               absolute disp32 (SIB form, no base)
    qword[0x100000000]          absolute 64 bit address, only for mov64
    qword[src + idx*8 + 16]     gp64 register holes as base/index (Fragment)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.errors import EncodeError
from ..core.labels import Label
from ..core.operand import Hole, Operand
from .regs import Reg, rip

_SIZE_NAMES = {1: "byte", 2: "word", 4: "dword", 8: "qword", 10: "tbyte", 16: "xmmword", 32: "ymmword"}


@dataclass(frozen=True, slots=True, repr=False)
class MemExpr(Operand):
    """`[base + index*scale + disp]`, or `[rip + label + disp]`.

    `size` is the access width in bytes, None when untyped (`ptr[...]`).
    Instances are normalized and validated on construction.
    """

    base: Reg | Hole | None = None  # a Hole is a gp64 register hole
    index: Reg | Hole | None = None
    scale: int = 1
    disp: int = 0
    label: Label | str | Hole | None = None  # str names a label of the assembler
    size: int | None = None

    def __post_init__(self):
        base, index, scale = self.base, self.index, self.scale
        if scale not in (1, 2, 4, 8):
            raise EncodeError(f"scale must be 1, 2, 4 or 8, got {scale!r}")
        # rsp can only be a base; with scale 1 swapping base and index is exact.
        if isinstance(index, Reg) and index.kind == "gp" and index.code == 4 and scale == 1:
            if base is None:
                base, index = index, None
            elif isinstance(base, Hole) or (base.kind == "gp" and base.code != 4):
                base, index = index, base
            object.__setattr__(self, "base", base)
            object.__setattr__(self, "index", index)
        for r in (base, index):
            if isinstance(r, Hole) and (r.kind != "reg" or r.regclass != "gp" or r.size != 8):
                raise EncodeError(f"{r!r} cannot be used in an address, only gp64 holes can")
        if isinstance(self.label, Hole) and self.label.kind != "label":
            raise EncodeError(f"{self.label!r} cannot be used as an address label")
        if isinstance(base, Reg):
            if base.kind not in ("gp", "rip") or (base.kind == "gp" and base.size not in (4, 8)):
                raise EncodeError(f"{base} cannot be a memory base")
        if isinstance(index, Hole):
            if isinstance(base, Reg) and base.kind == "gp" and base.size != 8:
                raise EncodeError(f"mixed address sizes: {base} and {index!r}")
        elif index is not None:
            if index.kind == "gp":
                if index.size not in (4, 8):
                    raise EncodeError(f"{index} cannot be a memory index")
                if index.code == 4:
                    raise EncodeError(f"{index} cannot be a memory index")
                if isinstance(base, Hole) and index.size != 8:
                    raise EncodeError(f"mixed address sizes: {base!r} and {index}")
                if isinstance(base, Reg) and base.kind == "gp" and base.size != index.size:
                    raise EncodeError(f"mixed address sizes: {base} and {index}")
            elif index.kind in ("xmm", "ymm"):
                raise EncodeError(f"VSIB addressing ({index} as index) is not supported")
            else:
                raise EncodeError(f"{index} cannot be a memory index")
        if base is rip and index is not None:
            raise EncodeError("rip-relative operands cannot have an index")
        if self.label is not None and base is not rip:
            raise EncodeError("labels are only allowed as rip-relative operands: [rip + label]")
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

    def _add_reg(self, reg: Reg, scale: int = 1) -> MemExpr:
        if scale == 1 and self.base is None:
            return replace(self, base=reg)
        if self.index is None:
            return replace(self, index=reg, scale=scale)
        raise EncodeError(f"too many registers in memory operand: {self} + {reg}")

    def __add__(self, other) -> MemExpr:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return replace(self, disp=self.disp + other)
        if isinstance(other, Hole):
            return self._add_hole(other)
        if isinstance(other, (Label, str)):
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

    __radd__ = __add__

    def _add_hole(self, hole: Hole) -> MemExpr:
        if hole.kind == "label":
            if self.label is not None:
                raise EncodeError("memory operand can reference only one label")
            return replace(self, label=hole)
        if hole.kind != "reg":
            raise EncodeError(f"{hole!r} cannot be part of an address")
        if self.base is rip:
            raise EncodeError(f"rip + {hole!r}: a register hole cannot be rip-relative")
        return self._add_reg(hole)

    def __sub__(self, other) -> MemExpr:
        if isinstance(other, int) and not isinstance(other, bool):
            return replace(self, disp=self.disp - other)
        return NotImplemented

    def __str__(self) -> str:
        parts = []
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


class SizePrefix:
    """`qword[...]` etc. Indexing produces a sized MemExpr."""

    __slots__ = ("name", "size")

    def __init__(self, name: str, size: int | None):
        self.name, self.size = name, size

    def __getitem__(self, x) -> MemExpr:
        if isinstance(x, MemExpr):
            m = x
        elif isinstance(x, Reg):
            m = MemExpr(base=x)
        elif isinstance(x, int) and not isinstance(x, bool):
            m = MemExpr(disp=x)
        elif isinstance(x, Hole) and x.kind == "reg":
            m = MemExpr(base=x)
        elif isinstance(x, (Label, str, Hole)):
            raise EncodeError(f"{self.name}[label] is not addressable on x64, use {self.name}[rip + label]")
        else:
            raise TypeError(f"{self.name}[...] expects a register, int or memory expression, got {x!r}")
        return replace(m, size=self.size)

    def __repr__(self) -> str:
        return self.name


byte = SizePrefix("byte", 1)
word = SizePrefix("word", 2)
dword = SizePrefix("dword", 4)
qword = SizePrefix("qword", 8)
tword = SizePrefix("tword", 10)
oword = SizePrefix("oword", 16)
yword = SizePrefix("yword", 32)
ptr = SizePrefix("ptr", None)

__all__ = ["MemExpr", "SizePrefix", "byte", "word", "dword", "qword", "tword", "oword", "yword", "ptr"]
