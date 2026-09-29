"""Typed memory operands: ctypes structures as views over a base address.

    p = typed(a0, Point)        # Point is a ctypes.Structure
    lw(a1, p.x)                 # Point.x.offset(a0), 4 bytes
    lbu(a2, p.tag)              # 1 byte
    fld(fa0, p.v[1])            # element of a ctypes array field
    ld(a3, typed(a0, Node).next)    # pointer fields are plain 8 byte scalars

A scalar field is a `MemExpr` that carries the field's size, and the
encoder rejects a load, store or atomic of another width (`ld a1, p.x`
with a 4 byte `x`). Offsets and sizes come from ctypes itself
(`Type.field.offset`, `ctypes.sizeof`), so `_pack_`, `_anonymous_`, unions
and inherited fields lay out exactly as ctypes lays them out. An offset
outside the signed 12 bit range is the encoder's usual out of range error.

riscv64 addresses are a base register plus an offset, with no index
register, so an array element is selected by a constant index only.
`view[a2]` raises EncodeError that shows how to compute the element's
address into a register (`sh2add` for a 4 byte element) and view the
element there.
"""

import ctypes
from dataclasses import replace
from typing import Any, cast, overload

from ..core.errors import EncodeError
from ..core.structs import (
    AGGREGATES,
    ArrayView,
    StructView,
    ctype_source,
    check_ctype,
    element_offset,
    get_field,
    scalar_size,
)
from .mem import Addr, MemExpr
from .regs import Reg, X, t0, t1, t2

__all__ = ["typed", "Typed", "TypedArray"]


@overload
def typed(base: X | Addr | MemExpr, ctype: type[ctypes.Structure] | type[ctypes.Union]) -> Typed: ...
@overload
def typed(base: X | Addr | MemExpr, ctype: type[ctypes.Array[Any]]) -> TypedArray: ...
@overload
def typed(base: X | Addr | MemExpr, ctype: type[object]) -> Any: ...
def typed(base: X | Addr | MemExpr, ctype: type) -> Any:
    """View the memory at `base` as the ctypes type `ctype`.

    `base` is an integer register (`a0`, `sp`), an address (`a0 + 16`) or
    an unsized `mem[...]`. A Structure or Union gives a `Typed` whose
    attributes are its fields, an Array gives a `TypedArray`, and a scalar
    type gives the sized memory operand directly.
    """
    m: MemExpr
    if isinstance(base, Reg):
        m = MemExpr(base)  # validated by MemExpr: an F register is not a base
    elif isinstance(base, Addr):
        m = MemExpr(cast(X, base.base), base.disp)
    elif isinstance(base, MemExpr):
        if base.size is not None:
            raise EncodeError(f"typed() needs an unsized address, got a {base.size} byte {base}")
        m = base
    else:
        raise TypeError(f"typed() base must be a register or address, got {base!r}")
    check_ctype(ctype)
    return _view(m, ctype)


def _view(m: MemExpr, ctype: type) -> Any:
    if issubclass(ctype, AGGREGATES):
        return Typed(m, ctype)
    if issubclass(ctype, ctypes.Array):
        return TypedArray(m, ctype)
    return replace(m, size=scalar_size(ctype, "riscv64"))


def _offset(m: MemExpr, offset: int) -> MemExpr:
    return replace(m, disp=m.disp + offset) if offset else m


def _element_address(m: MemExpr, i: Reg, esize: int) -> tuple[str, X]:
    """Code that puts the address of element `i` (without the offset of
    the array) in a temporary register, and that register."""
    base = m.base
    tmp = next(r for r in (t0, t1, t2) if r is not base and r is not i)
    if esize == 1:
        return f"add({tmp}, {base}, {i})", tmp
    if esize in (2, 4, 8):
        k = esize.bit_length() - 1
        return f"sh{k}add({tmp}, {i}, {base}) (Zba) or slli({tmp}, {i}, {k}) and add({tmp}, {base}, {tmp})", tmp
    return f"li({tmp}, {esize}), mul({tmp}, {i}, {tmp}) and add({tmp}, {base}, {tmp})", tmp


class Typed(StructView[MemExpr]):
    """A ctypes Structure or Union at a memory address.

    Attribute access yields the field: a sized `MemExpr` for scalars, a
    `Typed` for nested structures and unions, a `TypedArray` for arrays.
    Type checkers see fields as `Any`. `addr` is the unsized `MemExpr` of
    the start, `size` is `ctypes.sizeof(ctype)` and `ctype` the type. A
    field whose name clashes with these three is reached with
    `view["size"]`. Fields whose names start with an underscore (`_pad`)
    are attributes too; only dunder names are not fields.
    """

    __slots__ = ()

    def __getitem__(self, name: str) -> Any:
        field = get_field(self.ctype, name)
        return _view(_offset(self.addr, field.offset), field.type)


class TypedArray(ArrayView[MemExpr]):
    """A ctypes Array at a memory address.

    `view[i]` with an int is the element at a constant index: a sized
    `MemExpr`, a `Typed` or a nested `TypedArray` like a field. riscv64
    has no register indexed addressing, so `view[a2]` raises EncodeError
    with the code that computes the element address into a register.
    """

    __slots__ = ()

    def __getitem__(self, i: int) -> Any:
        elem = self.element
        name = self.ctype.__name__
        if isinstance(i, Reg):
            if not isinstance(i, X):
                raise EncodeError(f"{name}[{i}]: {i} cannot be an index register")
            m = self.addr
            esize = ctypes.sizeof(elem)
            d = m.disp
            if not esize:
                at = f"{m.base} + {d}" if d > 0 else f"{m.base} - {-d}" if d else str(m.base)
                raise EncodeError(
                    f"{name}[{i}]: the elements are 0 bytes, so every element is at the start of the "
                    f"array and the index selects nothing; view the element with typed({at}, {ctype_source(elem)})"
                )
            code, tmp = _element_address(m, i, esize)
            at = f"{tmp} + {d}" if d > 0 else f"{tmp} - {-d}" if d else str(tmp)
            raise EncodeError(
                f"{name}[{i}]: riscv64 loads and stores have no index register; put the element "
                f"address in a register first, {code}, and view the element with "
                f"typed({at}, {ctype_source(elem)})"
            )
        return _view(_offset(self.addr, element_offset(self.ctype, i)), elem)
