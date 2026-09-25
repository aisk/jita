"""Loading a linked image into executable memory and calling into it."""

from __future__ import annotations

import ctypes
import mmap
from collections.abc import Callable, Mapping
from typing import Any

from ..core.assembler import Assembler
from ..core.errors import LoadError
from ..core.labels import Label
from ..core.link import Image, layout, link
from .memory import ExecMemory


class Module:
    """Loaded code. Function objects returned by `function` keep the module
    alive through their `module` attribute, and the memory is released when
    the module is garbage collected, so `close` and `with` are optional.
    Calling a function after `close` crashes the process."""

    def __init__(self, image: Image, memory: ExecMemory):
        self.image = image
        self.memory = memory
        self.closed = False

    def _check_open(self) -> None:
        if self.closed or self.memory.closed:
            raise LoadError("module is closed")

    def address(self, label: Label | str) -> int:
        self._check_open()
        return self.image.address(label)

    def function(self, restype: Any, *argtypes: Any, entry: Label | str | None = None) -> Any:
        """ctypes CFUNCTYPE bound to `entry` (default: image base). Its
        `module` attribute is this module."""
        self._check_open()
        addr = self.image.base if entry is None else self.address(entry)
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
        fn.module = self
        return fn

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

    def __enter__(self) -> Module:
        return self

    def __exit__(self, *exc) -> None:
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
    return Module(image, mem)


def function(
    restype: Any,
    *argtypes: Any,
    entry: Label | str | None = None,
    externs: Mapping[str, int] | None = None,
    arch: Any = None,
) -> Callable[[Callable[[Assembler], Any]], Any]:
    """Decorator turning a code generator into a ctypes callable.

    The decorated body runs once, at decoration time, inside a fresh
    `Assembler(arch)` entered as the current context, and receives that
    assembler as its only argument. The result is
    `a.function(restype, *argtypes, entry=entry, externs=externs)`, with
    the body's `__name__`, `__qualname__` and `__doc__` copied onto it.
    Inside a factory function the body closes over the factory's
    parameters, which is how specialized variants are generated.
    """

    def decorate(body: Callable[[Assembler], Any]) -> Any:
        a = Assembler(arch)
        with a:
            body(a)
        fn = a.function(restype, *argtypes, entry=entry, externs=externs)
        for attr in ("__name__", "__qualname__", "__doc__"):
            try:
                setattr(fn, attr, getattr(body, attr))
            except (AttributeError, TypeError):
                pass
        return fn

    return decorate
