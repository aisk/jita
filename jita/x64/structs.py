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

import ctypes
from dataclasses import replace
from typing import Any, overload

from ..core.errors import EncodeError
from ..core.structs import (
    AGGREGATES,
    SCALAR_SIZES,
    ArrayView,
    StructView,
    check_ctype,
    element_offset,
    get_field,
    scalar_size,
)
from .mem import MemAny, MemExpr, sized
from .regs import Reg

__all__ = ["typed", "Typed", "TypedArray"]


@overload
def typed(base: Reg | MemExpr, ctype: type[ctypes.Structure] | type[ctypes.Union]) -> Typed: ...
@overload
def typed(base: Reg | MemExpr, ctype: type[ctypes.Array[Any]]) -> TypedArray: ...
@overload
def typed(base: Reg | MemExpr, ctype: type[object]) -> Any: ...
def typed(base: Reg | MemExpr, ctype: type) -> Any:
    """View the memory at `base` as the ctypes type `ctype`.

    `base` is a register (`rdi`) or an unsized memory expression
    (`rdi + 16`, `rip + "table"`). A Structure or Union gives a `Typed`
    whose attributes are its fields, an Array gives a `TypedArray`, and a
    scalar type gives the sized memory operand directly.
    """
    mem: MemAny
    if isinstance(base, Reg):
        mem = MemAny(base=base)
    elif isinstance(base, MemExpr):
        if base.size is not None:
            raise EncodeError(f"typed() needs an unsized address, got {base}; drop the size prefix")
        mem = base if isinstance(base, MemAny) else MemAny(base.base, base.index, base.scale, base.disp, base.label)
    else:
        raise TypeError(f"typed() base must be a register or memory expression, got {base!r}")
    check_ctype(ctype)
    return _view(mem, ctype)


def _view(mem: MemAny, ctype: type) -> Any:
    if issubclass(ctype, AGGREGATES):
        return Typed(mem, ctype)
    if issubclass(ctype, ctypes.Array):
        return TypedArray(mem, ctype)
    return _scalar(mem, ctype)


def _scalar(mem: MemAny, ctype: type) -> MemExpr:
    return sized(mem, scalar_size(ctype, "x64"))


def _offset(mem: MemAny, offset: int) -> MemAny:
    return mem + offset if offset else mem


class Typed(StructView[MemAny]):
    """A ctypes Structure or Union at a memory address.

    Attribute access yields the field: a sized `MemExpr` for scalars, a
    `Typed` for nested structures and unions, a `TypedArray` for arrays.
    Type checkers see fields as `Any`. `addr` is the unsized `MemAny` of
    the start, `size` is
    `ctypes.sizeof(ctype)` and `ctype` the type. A field whose name clashes
    with these three is reached with `view["size"]`. Fields whose names
    start with an underscore (`_pad`) are attributes too; only dunder
    names are not fields.
    """

    __slots__ = ()

    def __getitem__(self, name: str) -> Any:
        field = get_field(self.ctype, name)
        return _view(_offset(self.addr, field.offset), field.type)


class TypedArray(ArrayView[MemAny]):
    """A ctypes Array at a memory address.

    `view[i]` with an int is the element at a constant index, `view[reg]`
    the element at a register index (the element size must be 1, 2, 4 or 8
    and the address must not already have an index register). Elements are
    sized `MemExpr`s, `Typed` or nested `TypedArray` views like fields.
    """

    __slots__ = ()

    def __getitem__(self, i: int | Reg) -> Any:
        elem = self.element
        esize = ctypes.sizeof(elem)
        if isinstance(i, Reg):
            if esize not in SCALAR_SIZES:
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
        return _view(_offset(self.addr, element_offset(self.ctype, i)), elem)
