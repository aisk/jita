"""Loading a linked image into executable memory and calling into it."""

import ctypes
import inspect
import mmap
import types
from collections.abc import Callable, Mapping
from typing import Any, Protocol, Self, cast

from ..core.assembler import Assembler
from ..core.errors import LoadError
from ..core.labels import Label
from ..core.link import Image, layout, link
from .memory import ExecMemory


# Calling conventions of `Module.builtin`, the `METH_*` flags of CPython's
# PyMethodDef. They are part of the stable ABI.
METH_VARARGS = 0x0001
METH_KEYWORDS = 0x0002
METH_NOARGS = 0x0004
METH_O = 0x0008
METH_FASTCALL = 0x0080

_CALLS = {
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
    METH_VARARGS | METH_KEYWORDS,
    METH_FASTCALL,
    METH_FASTCALL | METH_KEYWORDS,
}


class _MethodDef(ctypes.Structure):
    _fields_ = (
        ("ml_name", ctypes.c_char_p),
        ("ml_meth", ctypes.c_void_p),
        ("ml_flags", ctypes.c_int),
        ("ml_doc", ctypes.c_char_p),
    )


# Every PyMethodDef ever handed to CPython, by content. A builtin reads its
# PyMethodDef while it is deallocated, after a cyclic GC may already have
# cleared everything else it refers to, so no Python object can own one.
# They are never freed; equal content shares one, which bounds the table
# when code is regenerated at reused addresses.
_method_defs: dict[tuple[int, int, str, str | None], _MethodDef] = {}


# A private prototype, so the shared ctypes.pythonapi.PyCMethod_New keeps
# its default argtypes.
_PyCMethod_New = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.c_void_p
)(("PyCMethod_New", ctypes.pythonapi))


class CFunction(Protocol):
    """What `function` returns: a ctypes function pointer into loaded
    code. Its Python signature comes from the ctypes argument types at
    runtime and is not known statically."""

    module: Module
    __name__: str
    __qualname__: str
    __doc__: str | None

    def __call__(self, *args: Any) -> Any: ...


class Builtin(Protocol):
    """What `builtin` returns: a builtin function whose C implementation
    is the loaded code. `__self__` is the module, passed to the code as
    its `self` argument."""

    __self__: Module
    __name__: str
    __qualname__: str
    __doc__: str | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class Module(types.ModuleType):
    """Loaded code. Functions returned by `function` and `builtin` keep the
    module alive, and the memory is released when the module is garbage
    collected, so `close` and `with` are optional. Calling a function after
    `close` crashes the process.

    It is a module object so that CPython names a builtin whose `self` it
    is like a module level function (`f()`, not `Module.f()`)."""

    def __init__(self, image: Image, memory: ExecMemory, assembler: Assembler):
        super().__init__("jita.Module", type(self).__doc__)
        self.image = image
        self.memory = memory
        self.assembler = assembler
        self.closed = False

    def __repr__(self) -> str:
        return f"<jita.Module at {self.image.base:#x}>"

    def __dir__(self) -> list[str]:
        # ModuleType lists only the instance dict.
        return sorted({*super().__dir__(), *dir(type(self))})

    def _check_open(self) -> None:
        if self.closed or self.memory.closed:
            raise LoadError("module is closed")

    def address(self, label: Label | str) -> int:
        self._check_open()
        return self.image.address(label)

    def _entry(self, entry: Label | str | None) -> int:
        self._check_open()
        return self.image.base if entry is None else self.address(entry)

    def function(self, restype: Any, *argtypes: Any, entry: Label | str | None = None) -> CFunction:
        """A `ctypes.CFUNCTYPE(restype, *argtypes)` pointer to `entry`
        (default: image base). The GIL is released during the call. Its
        `module` attribute is this module."""
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(self._entry(entry))
        setattr(fn, "module", self)
        return cast(CFunction, fn)

    def builtin(
        self,
        flags: int,
        *,
        entry: Label | str | None = None,
        name: str | None = None,
        doc: str | None = None,
        module_name: str | None = None,
    ) -> Builtin:
        """A builtin function implemented by the code at `entry` (default:
        image base) with the calling convention `flags`, one of METH_NOARGS,
        METH_O, METH_VARARGS, METH_FASTCALL, the last two optionally with
        METH_KEYWORDS. The code is called with the GIL held and this module
        as `self`. `name` defaults to the name of `entry`, `module_name`
        becomes `__module__`."""
        if type(flags) is not int or flags not in _CALLS:
            raise ValueError(f"unsupported builtin calling convention {flags!r}")
        if name is None:
            name = entry if isinstance(entry, str) else (entry.name if entry is not None else None)
        key = (self._entry(entry), flags, name or "builtin", doc)
        md = _method_defs.get(key)
        if md is None:
            md = _MethodDef(key[2].encode(), key[0], flags, None if doc is None else doc.encode())
            md = _method_defs.setdefault(key, md)
        return cast(Builtin, _PyCMethod_New(ctypes.addressof(md), self, module_name, None))

    def write(self, where: int | Label | str, data: bytes) -> None:
        """Overwrite loaded bytes at `where`, an offset from the image base
        or a label/symbol name. Only writable sections accept writes; the
        executable prefix raises LoadError."""
        self._check_open()
        offset = where if isinstance(where, int) else self.image.address(where) - self.image.base
        if offset < 0 or offset + len(data) > len(self.image.data):
            raise LoadError(
                f"write of {len(data)} bytes at {offset:#x} is outside the image "
                f"({len(self.image.data):#x} bytes)"
            )
        self.memory.write(data, offset)

    def close(self) -> None:
        self.closed = True
        self.memory.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load(asm: Assembler, externs: Mapping[str, int] | None = None) -> Module:
    """Allocate memory, link at its real address, write and protect it."""
    page = mmap.PAGESIZE
    for sec in asm.sections.values():
        if sec.align > page:
            raise LoadError(
                f"section {sec.name!r} alignment {sec.align} exceeds the page size {page}"
            )
    size = layout(asm, page).size
    mem = ExecMemory(size, icache_flush=asm.arch.icache_flush)
    try:
        image = link(asm, mem.address, externs)
        mem.write(image.data)
        mem.protect_exec(image.exec_size)
    except BaseException:
        mem.close()
        raise
    return Module(image, mem, asm)


def _generate(body: Callable[..., object], arch: object) -> Assembler:
    a = Assembler(arch)
    with a:
        if inspect.signature(body).parameters:
            body(a)
        else:
            body()
    return a


def function(
    restype: Any,
    *argtypes: Any,
    entry: Label | str | None = None,
    externs: Mapping[str, int] | None = None,
    arch: object = None,
) -> Callable[[Callable[..., object]], CFunction]:
    """Decorator turning a code generator into a ctypes callable.

    The decorated body runs once, at decoration time, inside a fresh
    `Assembler(arch)` entered as the current context. A body that takes a
    parameter receives that assembler (for `a.pc`, `a.section`, ...); a
    body without parameters is called with none. The result is
    `a.function(restype, *argtypes, entry=entry, externs=externs)`, with
    the body's `__name__`, `__qualname__` and `__doc__` copied onto it.
    Inside a factory function the body closes over the factory's
    parameters, which is how specialized variants are generated.
    """

    def decorate(body: Callable[..., object]) -> CFunction:
        fn = _generate(body, arch).function(restype, *argtypes, entry=entry, externs=externs)
        for attr in ("__name__", "__qualname__", "__doc__"):
            try:
                setattr(fn, attr, getattr(body, attr))
            except (AttributeError, TypeError):
                pass
        return fn

    return decorate


def builtin(
    flags: int,
    *,
    entry: Label | str | None = None,
    externs: Mapping[str, int] | None = None,
    arch: object = None,
) -> Callable[[Callable[..., object]], Builtin]:
    """Decorator turning a code generator into a builtin function. The body
    runs as in `function`; the result is `a.builtin(flags, entry=entry,
    externs=externs)` named after the body, with its `__doc__` and
    `__module__`."""

    def decorate(body: Callable[..., object]) -> Builtin:
        return _generate(body, arch).builtin(
            flags,
            entry=entry,
            externs=externs,
            name=body.__name__,
            doc=body.__doc__,
            module_name=body.__module__,
        )

    return decorate
