"""Typed memory operands: ctypes structures as views over a base address.

    p = typed(a0, Point)        # Point is a ctypes.Structure
    ld.w(a1, p.x)               # ld.w $a1, $a0, Point.x.offset, 4 bytes
    ld.bu(a2, p.tag)            # 1 byte
    fld.d(fa0, p.v[1])          # element of a ctypes array field
    ldx.bu(a3, typed(a1, Bytes)[a2])    # ldx.bu $a3, $a1, $a2
    ld.d(a4, typed(a0, Node).next)      # pointer fields are plain 8 byte scalars

A field or element is a `MemExpr` that stands for the base and offset
operands of the instruction (`ld.w(a1, p.x)` is `ld.w $a1, $a0, 0`) and
carries the field's size; the encoder rejects a load, store or atomic of
another width. Offsets and sizes come from ctypes itself
(`Type.field.offset`, `ctypes.sizeof`), so `_pack_`, `_anonymous_`, unions
and inherited fields lay out exactly as ctypes lays them out. An offset
outside the instruction's range is the encoder's usual out of range error.

The indexed loads and stores (`ldx.*`, `stx.*`, `fldx.*`, `fstx.*`) add an
index register to the base without scaling it and take no offset. So a
register index needs an array of 1 byte elements that starts at the base
(`typed(a1, Bytes)[a2]`, not `p.name[a2]` at a nonzero offset). Other
element sizes raise EncodeError that shows how to compute the element's
address into a register (`alsl.d` for a 2, 4, 8 or 16 byte element) and
view the element there.
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
from .regs import R, Reg, t0, t1, t2

__all__ = ["typed", "Typed", "TypedArray"]


@overload
def typed(base: R | Addr | MemExpr, ctype: type[ctypes.Structure] | type[ctypes.Union]) -> Typed: ...
@overload
def typed(base: R | Addr | MemExpr, ctype: type[ctypes.Array[Any]]) -> TypedArray: ...
@overload
def typed(base: R | Addr | MemExpr, ctype: type[object]) -> Any: ...
def typed(base: R | Addr | MemExpr, ctype: type) -> Any:
    """View the memory at `base` as the ctypes type `ctype`.

    `base` is an integer register (`a0`, `sp`), an address (`a0 + 16`) or
    the unsized `addr` of another view. A Structure or Union gives a
    `Typed` whose attributes are its fields, an Array gives a
    `TypedArray`, and a scalar type gives the sized memory operand
    directly.
    """
    m: MemExpr
    if isinstance(base, Reg):
        m = MemExpr(base)  # validated by MemExpr: an F register is not a base
    elif isinstance(base, Addr):
        m = MemExpr(cast(R, base.base), base.disp)
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
    return replace(m, size=scalar_size(ctype, "loongarch64"))


def _offset(m: MemExpr, offset: int, what: str) -> MemExpr:
    if not offset:
        return m
    if m.index is not None:
        raise EncodeError(
            f"{what} is {offset} bytes past {m}, and a loongarch64 address cannot have both an index "
            f"register and an offset; add.d the index to the base into a register first and view "
            "the element there"
        )
    return replace(m, disp=m.disp + offset)


def _element_hint(name: str, m: MemExpr, i: Reg, elem: type, why: str) -> EncodeError:
    """The error for `view[i]` that `ldx` cannot address, with the code
    that computes the element address into a register instead."""
    base, esize = m.base, ctypes.sizeof(elem)
    tmp = next(r for r in (t0, t1, t2) if r is not base and r is not i)
    if esize == 1:
        code = f"add.d({tmp.name}, {base.name}, {i.name})"
    elif esize in (2, 4, 8, 16):
        code = f"alsl.d({tmp.name}, {i.name}, {base.name}, {esize.bit_length() - 1})"
    else:
        t = tmp.name
        code = f"li.d({t}, {esize}), mul.d({t}, {i.name}, {t}) and add.d({t}, {base.name}, {t})"
    d = m.disp
    at = f"{tmp.name} + {d}" if d > 0 else f"{tmp.name} - {-d}" if d else tmp.name
    return EncodeError(
        f"{name}[{i.name}]: {why}; put the element address in a register first, {code}, and view "
        f"the element with typed({at}, {ctype_source(elem)})"
    )


def _zero_size(name: str, i: Reg, m: MemExpr, elem: type) -> str:
    """The error for `view[i]` when the elements have no size: every
    element is at the array's own address, so there is nothing to index."""
    d = m.disp
    at = f"{m.base.name} + {d}" if d > 0 else f"{m.base.name} - {-d}" if d else m.base.name
    return (
        f"{name}[{i.name}]: the elements are 0 bytes, so every element is at the start of the array "
        f"and the index selects nothing; view the element with typed({at}, {ctype_source(elem)})"
    )


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
        return _view(_offset(self.addr, field.offset, f"{self.ctype.__name__}.{name}"), field.type)


class TypedArray(ArrayView[MemExpr]):
    """A ctypes Array at a memory address.

    `view[i]` with an int is the element at a constant index. `view[a2]`
    is the element at a register index, the base and index operands of
    `ldx.*`, `stx.*`, `fldx.*` and `fstx.*`, which do not scale the index:
    it needs 1 byte elements and an address without an offset or another
    index. Elements are sized `MemExpr`s, `Typed` or nested `TypedArray`
    views like fields.
    """

    __slots__ = ()

    def __getitem__(self, i: int | R) -> Any:
        elem = self.element
        name = self.ctype.__name__
        if isinstance(i, Reg):
            m = self.addr
            if not isinstance(i, R):
                raise EncodeError(f"{name}[{i}]: {i} cannot be an index register")
            if m.index is not None:
                raise EncodeError(f"{name}[{i.name}]: {m} already has an index register")
            esize = ctypes.sizeof(elem)
            if not esize:
                raise EncodeError(_zero_size(name, i, m, elem))
            if esize != 1:
                why = f"ldx adds the index unscaled, and the elements are {esize} bytes"
                raise _element_hint(name, m, i, elem, why)
            if m.disp:
                raise _element_hint(
                    name, m, i, elem, f"the array is at {m}, and ldx takes no offset besides the index register"
                )
            return _view(MemExpr(m.base, index=i), elem)
        return _view(_offset(self.addr, element_offset(self.ctype, i), f"{name}[{i}]"), elem)
