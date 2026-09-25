"""Typed memory operands: ctypes structures as views over a base address.

    p = typed(rdi, Point)       # Point is a ctypes.Structure
    mov(eax, p.x)               # dword[rdi + Point.x.offset]
    movzx(eax, p.tag)           # byte[...]
    movsd(xmm0, p.v[1])         # element of a ctypes array field
    movsd(xmm1, p.v[rcx])       # register index, scaled by the element size
    lea(rax, p.v.addr)          # unsized address of the field
    typed(rax, Node).next       # pointer fields are plain qword scalars

Offsets and sizes come from ctypes itself (`Type.field.offset`,
`ctypes.sizeof`), so `_pack_`, `_anonymous_`, unions and inherited fields
lay out exactly as ctypes lays them out.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

from ..core.errors import EncodeError
from .mem import MemExpr
from .regs import Reg

__all__ = ["typed", "Typed", "TypedArray"]

_AGGREGATES = (ctypes.Structure, ctypes.Union)
_SCALARS = (ctypes._SimpleCData, ctypes._Pointer, ctypes._CFuncPtr)
_SCALAR_SIZES = (1, 2, 4, 8)


def typed(base: Reg | MemExpr, ctype: type) -> Any:
    """View the memory at `base` as the ctypes type `ctype`.

    `base` is a register (`rdi`) or an unsized memory expression
    (`rdi + 16`, `rip + "table"`). A Structure or Union gives a `Typed`
    whose attributes are its fields, an Array gives a `TypedArray`, and a
    scalar type gives the sized memory operand directly.
    """
    if isinstance(base, Reg):
        mem = MemExpr(base=base)
    elif isinstance(base, MemExpr):
        if base.size is not None:
            raise EncodeError(f"typed() needs an unsized address, got {base}; drop the size prefix")
        mem = base
    else:
        raise TypeError(f"typed() base must be a register or memory expression, got {base!r}")
    if not isinstance(ctype, type) or not issubclass(ctype, (*_AGGREGATES, ctypes.Array, *_SCALARS)):
        raise TypeError(f"typed() expects a ctypes type, got {ctype!r}")
    return _view(mem, ctype)


def _view(mem: MemExpr, ctype: type) -> Any:
    if issubclass(ctype, _AGGREGATES):
        return Typed(mem, ctype)
    if issubclass(ctype, ctypes.Array):
        return TypedArray(mem, ctype)
    return _scalar(mem, ctype)


def _scalar(mem: MemExpr, ctype: type) -> MemExpr:
    if getattr(ctype, "_type_", None) == "g":
        raise EncodeError(f"{ctype.__name__} (long double) has no x64 memory operand size")
    le = getattr(ctype, "__ctype_le__", ctype)
    if le is not ctype:
        raise EncodeError(f"{ctype.__name__} is a big endian field; x64 loads are little endian")
    size = ctypes.sizeof(ctype)
    if size not in _SCALAR_SIZES:
        raise EncodeError(f"{ctype.__name__} is {size} bytes, not a 1, 2, 4 or 8 byte scalar")
    return replace(mem, size=size)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _offset(mem: MemExpr, offset: int) -> MemExpr:
    return mem + offset if offset else mem


class Typed:
    """A ctypes Structure or Union at a memory address.

    Attribute access yields the field: a sized `MemExpr` for scalars, a
    `Typed` for nested structures and unions, a `TypedArray` for arrays.
    `addr` is the unsized `MemExpr` of the start, `size` is
    `ctypes.sizeof(ctype)` and `ctype` the type. A field whose name clashes
    with these three is reached with `view["size"]`. Fields whose names
    start with an underscore (`_pad`) are attributes too; only dunder
    names are not fields.
    """

    __slots__ = ("addr", "ctype")

    def __init__(self, addr: MemExpr, ctype: type):
        object.__setattr__(self, "addr", addr)
        object.__setattr__(self, "ctype", ctype)

    @property
    def size(self) -> int:
        return ctypes.sizeof(self.ctype)

    def _fields(self) -> list[str]:
        t = self.ctype
        found = [(f.offset, n) for n in dir(t) if isinstance(f := getattr(t, n, None), ctypes.CField)]
        return [n for _, n in sorted(found)]

    def __getitem__(self, name: str) -> Any:
        if not isinstance(name, str):
            raise TypeError(f"{self.ctype.__name__} fields are selected by name, got {name!r}")
        field = getattr(self.ctype, name, None) if not _is_dunder(name) else None
        if not isinstance(field, ctypes.CField):
            raise AttributeError(
                f"{self.ctype.__name__} has no field {name!r}; fields: {', '.join(self._fields())}"
            )
        if field.is_bitfield:
            raise EncodeError(f"{self.ctype.__name__}.{name} is a bit field, which has no memory operand")
        return _view(_offset(self.addr, field.offset), field.type)

    def __getattr__(self, name: str) -> Any:
        if _is_dunder(name):
            raise AttributeError(name)
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __dir__(self) -> list[str]:
        return list(dict.fromkeys(["addr", "ctype", "size", *self._fields()]))

    def __repr__(self) -> str:
        return f"typed({self.addr}, {self.ctype.__name__})"


class TypedArray:
    """A ctypes Array at a memory address.

    `view[i]` with an int is the element at a constant index, `view[reg]`
    the element at a register index (the element size must be 1, 2, 4 or 8
    and the address must not already have an index register). Elements are
    sized `MemExpr`s, `Typed` or nested `TypedArray` views like fields.
    """

    __slots__ = ("addr", "ctype")

    def __init__(self, addr: MemExpr, ctype: type):
        object.__setattr__(self, "addr", addr)
        object.__setattr__(self, "ctype", ctype)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only")

    @property
    def size(self) -> int:
        return ctypes.sizeof(self.ctype)

    @property
    def element(self) -> type:
        return self.ctype._type_

    def __len__(self) -> int:
        return self.ctype._length_

    def __getitem__(self, i: int | Reg) -> Any:
        elem = self.element
        esize = ctypes.sizeof(elem)
        if isinstance(i, Reg):
            if esize not in _SCALAR_SIZES:
                raise EncodeError(
                    f"{self.ctype.__name__}[{i}]: element size {esize} is not a valid scale (1, 2, 4, 8)"
                )
            if self.addr.index is not None:
                raise EncodeError(f"{self.ctype.__name__}[{i}]: {self.addr} already has an index register")
            if self.addr.base is None:
                mem = replace(self.addr, index=i, scale=esize)
            else:
                mem = self.addr._add_reg(i, esize)
            return _view(mem, elem)
        if isinstance(i, int) and not isinstance(i, bool):
            n = len(self)
            if n and not 0 <= i < n:
                raise IndexError(f"{self.ctype.__name__} index {i} out of range 0..{n - 1}")
            return _view(_offset(self.addr, i * esize), elem)
        raise TypeError(f"{self.ctype.__name__} index must be an int or a register, got {i!r}")

    def __repr__(self) -> str:
        return f"typed({self.addr}, {self.ctype.__name__})"
