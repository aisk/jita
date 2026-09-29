"""Architecture independent parts of `typed()`: ctypes introspection and
the view classes the architectures build their `Typed` and `TypedArray` on.

An architecture supplies the address type `A` and the `__getitem__` of
both views, which turn a field or an element into an address and then
into a sized memory operand, a nested `Typed` or a `TypedArray`.
"""

import ctypes
from _ctypes import CFuncPtr
from typing import Any

from .errors import EncodeError

__all__ = [
    "AGGREGATES",
    "SCALARS",
    "SCALAR_SIZES",
    "ArrayView",
    "StructView",
    "check_ctype",
    "element_offset",
    "get_field",
    "is_dunder",
    "scalar_size",
]

AGGREGATES = (ctypes.Structure, ctypes.Union)
SCALARS = (ctypes._SimpleCData, ctypes._Pointer, CFuncPtr)
SCALAR_SIZES = (1, 2, 4, 8)


def check_ctype(ctype: object) -> None:
    """TypeError unless `ctype` is a ctypes structure, union, array or scalar type."""
    if not isinstance(ctype, type) or not issubclass(ctype, (*AGGREGATES, ctypes.Array, *SCALARS)):
        raise TypeError(f"typed() expects a ctypes type, got {ctype!r}")


def scalar_size(ctype: type, arch: str) -> int:
    """The access size of a scalar ctypes type: 1, 2, 4 or 8.

    Long double, big endian types and other sizes have no memory operand
    on the little endian architectures jita supports.
    """
    if getattr(ctype, "_type_", None) == "g":
        raise EncodeError(f"{ctype.__name__} (long double) has no {arch} memory operand size")
    le = getattr(ctype, "__ctype_le__", ctype)
    if le is not ctype:
        raise EncodeError(f"{ctype.__name__} is a big endian field; {arch} loads are little endian")
    size = ctypes.sizeof(ctype)
    if size not in SCALAR_SIZES:
        raise EncodeError(f"{ctype.__name__} is {size} bytes, not a 1, 2, 4 or 8 byte scalar")
    return size


def is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _field_names(ctype: type) -> list[str]:
    found = [(f.offset, n) for n in dir(ctype) if isinstance(f := getattr(ctype, n, None), ctypes.CField)]
    return [n for _, n in sorted(found)]


def get_field(ctype: type, name: str) -> Any:
    """The `ctypes.CField` of `ctype` called `name`, checked to have a
    memory operand."""
    if not isinstance(name, str):
        raise TypeError(f"{ctype.__name__} fields are selected by name, got {name!r}")
    field = getattr(ctype, name, None) if not is_dunder(name) else None
    if not isinstance(field, ctypes.CField):
        raise AttributeError(f"{ctype.__name__} has no field {name!r}; fields: {', '.join(_field_names(ctype))}")
    if field.is_bitfield:
        raise EncodeError(f"{ctype.__name__}.{name} is a bit field, which has no memory operand")
    return field


def element_offset(ctype: type[ctypes.Array[Any]], i: object) -> int:
    """The byte offset of the element of `ctype` at the constant index `i`.
    A zero length array (a flexible array member) is not bounds checked."""
    if isinstance(i, int) and not isinstance(i, bool):
        n = getattr(ctype, "_length_")
        if n and not 0 <= i < n:
            raise IndexError(f"{ctype.__name__} index {i} out of range 0..{n - 1}")
        return i * ctypes.sizeof(getattr(ctype, "_type_"))
    raise TypeError(f"{ctype.__name__} index must be an int or a register, got {i!r}")


class StructView[A]:
    """A ctypes Structure or Union at the address `addr`.

    `view[name]` and `view.name` are the fields; the architecture's
    `__getitem__` turns `get_field(self.ctype, name)` into an operand or
    view. Helpers are module functions, not methods, so they do not
    shadow field names.
    """

    __slots__ = ("addr", "ctype")
    addr: A
    ctype: type

    def __init__(self, addr: A, ctype: type):
        object.__setattr__(self, "addr", addr)
        object.__setattr__(self, "ctype", ctype)

    @property
    def size(self) -> int:
        return ctypes.sizeof(self.ctype)

    def __getitem__(self, name: str) -> Any:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        if is_dunder(name):
            raise AttributeError(name)
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __dir__(self) -> list[str]:
        return list(dict.fromkeys(["addr", "ctype", "size", *_field_names(self.ctype)]))

    def __repr__(self) -> str:
        return f"typed({self.addr}, {self.ctype.__name__})"


class ArrayView[A]:
    """A ctypes Array at the address `addr`. The architecture's
    `__getitem__` handles register indexes; `element_offset` checks a
    constant index and gives its byte offset."""

    __slots__ = ("addr", "ctype")
    addr: A
    ctype: type[ctypes.Array[Any]]

    def __init__(self, addr: A, ctype: type[ctypes.Array[Any]]):
        object.__setattr__(self, "addr", addr)
        object.__setattr__(self, "ctype", ctype)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only")

    @property
    def size(self) -> int:
        return ctypes.sizeof(self.ctype)

    @property
    def element(self) -> type:
        return getattr(self.ctype, "_type_")

    def __len__(self) -> int:
        return getattr(self.ctype, "_length_")

    def __getitem__(self, i: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"typed({self.addr}, {self.ctype.__name__})"
