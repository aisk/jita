"""Typed memory operands: ctypes structures as views over a base address.

    p = typed(x0, Point)        # Point is a ctypes.Structure
    ldr(w1, p.x)                # [x0, #Point.x.offset], 4 bytes
    ldrb(w2, p.tag)             # 1 byte
    ldr(d0, p.v[1])             # element of a ctypes array field
    ldr(d1, typed(x2, Vec)[x3]) # [x2, x3, lsl #3], index scaled by the element size
    ldr(x4, typed(x0, Node).next)   # pointer fields are plain 8 byte scalars

A scalar field is a `MemExpr` that carries the field's size, and the
encoder rejects a load or store of another width (`ldr x1, p.x` with a
4 byte `x`). Offsets and sizes come from ctypes itself (`Type.field.offset`,
`ctypes.sizeof`), so `_pack_`, `_anonymous_`, unions and inherited fields
lay out exactly as ctypes lays them out.

aarch64 addresses are `[base, #offset]` or `[base, index{, lsl #s}]`, never
both, and the index shift is 0 or the access size. So a register index
needs an array that starts at the base (`typed(x2, Vec)[x3]`, not
`p.v[x3]` at a nonzero offset) and a scalar that fills the element (the
element itself, or a field at offset 0 as wide as the element). Other
combinations raise EncodeError; add the offset to a register first.

ldp and stp access two slots, so a typed operand carries its extent, the
bytes from it to the end of its array (`MemExpr.extent`). A pair must be
two elements of one array (`ldp(w1, w2, p.v[2])` needs a `p.v[3]`); a
field outside an array is one slot and cannot start a pair, even when a
field of the same size follows it. The elements of a zero length array
are unbounded. Register indexed elements never reach a pair, as ldp/stp
take no index register.
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
from .mem import Addr, MemExpr
from .regs import Mod, Reg, RegMod

__all__ = ["typed", "Typed", "TypedArray"]


@overload
def typed(base: Reg | Addr | MemExpr, ctype: type[ctypes.Structure] | type[ctypes.Union]) -> Typed: ...
@overload
def typed(base: Reg | Addr | MemExpr, ctype: type[ctypes.Array[Any]]) -> TypedArray: ...
@overload
def typed(base: Reg | Addr | MemExpr, ctype: type[object]) -> Any: ...
def typed(base: Reg | Addr | MemExpr, ctype: type) -> Any:
    """View the memory at `base` as the ctypes type `ctype`.

    `base` is a register (`x0`, `sp`), an address (`x0 + 16`) or an
    unsized `mem[...]`. A Structure or Union gives a `Typed` whose
    attributes are its fields, an Array gives a `TypedArray`, and a scalar
    type gives the sized memory operand directly.
    """
    m: MemExpr
    if isinstance(base, Reg):
        m = MemExpr(base)
    elif isinstance(base, Addr):
        m = MemExpr(base.base, base.index, base.mod, base.disp)
    elif isinstance(base, MemExpr):
        if base.size is not None:
            raise EncodeError(f"typed() needs an unsized address, got a {base.size} byte {base}")
        if base.mode != "offset":
            raise EncodeError(f"typed() needs a plain address, not the {base.mode}-index {base}")
        m = base
    else:
        raise TypeError(f"typed() base must be a register or address, got {base!r}")
    check_ctype(ctype)
    return _view(m, ctype, ctypes.sizeof(ctype))


def _view(m: MemExpr, ctype: type, extent: int | None) -> Any:
    """`extent` is the room from `m` to the end of the enclosing array or
    field, None when unbounded; only a scalar keeps it."""
    if issubclass(ctype, AGGREGATES):
        return Typed(m, ctype)
    if issubclass(ctype, ctypes.Array):
        return TypedArray(m, ctype)
    return _scalar(m, ctype, extent)


def _scalar(m: MemExpr, ctype: type, extent: int | None) -> MemExpr:
    size = scalar_size(ctype, "aarch64")
    shift = 0 if m.mod is None or m.mod.amount is None else m.mod.amount
    if m.index is not None and shift and 1 << shift != size:
        raise EncodeError(
            f"{ctype.__name__} at {m}: the index is scaled by {1 << shift}, but a {size} byte "
            f"access can only scale it by {size}; add the scaled index to the base register first "
            "and view the element there"
        )
    return replace(m, size=size, extent=extent)


def _offset(m: MemExpr, offset: int, what: str) -> MemExpr:
    if not offset:
        return m
    if m.index is not None:
        raise EncodeError(
            f"{what} is {offset} bytes past {m}, and an aarch64 address cannot have both an index "
            "register and an offset; add the index to the base register first"
        )
    return replace(m, disp=m.disp + offset)


def _index(i: Reg | RegMod[Any], shift: int, what: str) -> tuple[Reg, Mod | None]:
    """The index register and modifier for an element of 1 << shift bytes.

    An x index is plain, `lsl` or `sxtx`, a w index `uxtw` or `sxtw`; an
    amount, if written, must be the element size's shift.
    """
    reg = i.reg if isinstance(i, RegMod) else i
    if not isinstance(reg, Reg) or reg.kind != "gp":
        raise EncodeError(f"{what}: {reg} cannot be an index register")
    wide = reg.rt == "x"
    ok = f"{reg} or {reg}.sxtx()" if wide else f"{reg}.uxtw() or {reg}.sxtw()"
    if isinstance(i, Reg):
        if not wide:
            raise EncodeError(f"{what}: a 32 bit index needs an extend, {ok}")
        return i, Mod("lsl", shift) if shift else None
    kind, amount = i.mod.kind, i.mod.amount
    if kind not in (("lsl", "sxtx") if wide else ("uxtw", "sxtw")):
        raise EncodeError(f"{what}: a {64 if wide else 32} bit index cannot take {kind}; write {ok}")
    if amount is not None and amount != shift:
        plain = str(reg) if kind == "lsl" else f"{reg}.{kind}()"
        raise EncodeError(
            f"{what}: the element size scales the index by {1 << shift}, not by {1 << amount}; "
            f"write {plain} and let the element size set the shift"
        )
    if kind == "lsl":
        return reg, Mod("lsl", shift) if shift else None
    return reg, Mod(kind, shift if shift else None)


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
        where = _offset(self.addr, field.offset, f"{self.ctype.__name__}.{name}")
        return _view(where, field.type, ctypes.sizeof(field.type))


class TypedArray(ArrayView[MemExpr]):
    """A ctypes Array at a memory address.

    `view[i]` with an int is the element at a constant index. `view[x3]`
    is the element at a register index, `[base, x3, lsl #s]` with the
    element size as the scale; a w register needs an extend
    (`view[w3.uxtw()]`, `view[w3.sxtw()]`). A register index needs an
    element size of 1, 2, 4 or 8, an address without an offset or another
    index, and element accesses of the element size. Elements are sized
    `MemExpr`s, `Typed` or nested `TypedArray` views like fields. A scalar
    element at a constant index reaches to the end of the array, so
    `ldp(w1, w2, view[i])` works while `view[i + 1]` exists.
    """

    __slots__ = ()

    def __getitem__(self, i: int | Reg | RegMod[Any]) -> Any:
        elem = self.element
        name = self.ctype.__name__
        if isinstance(i, (Reg, RegMod)):
            esize = ctypes.sizeof(elem)
            if esize not in SCALAR_SIZES:
                raise EncodeError(f"{name}[{i}]: element size {esize} is not a valid scale (1, 2, 4, 8)")
            m = self.addr
            if m.index is not None:
                raise EncodeError(f"{name}[{i}]: {m} already has an index register")
            if m.disp:
                raise EncodeError(
                    f"{name}[{i}]: the array is at {m}, and an aarch64 address cannot have both an "
                    "index register and an offset; add the offset to the base register first"
                )
            reg, mod = _index(i, esize.bit_length() - 1, f"{name}[{i}]")
            return _view(MemExpr(m.base, reg, mod), elem, esize)
        offset = element_offset(self.ctype, i)
        rest = self.size - offset if len(self) else None
        return _view(_offset(self.addr, offset, f"{name}[{i}]"), elem, rest)
