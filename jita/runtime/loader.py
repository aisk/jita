"""Loading a linked image into executable memory and calling into it."""

from __future__ import annotations

import ctypes
import mmap
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..core.errors import LoadError
from ..core.label import Label
from ..core.link import Image, layout, link
from .memory import ExecMemory

if TYPE_CHECKING:
    from ..core.assembler import Assembler


class Module:
    """Loaded code. Function objects returned by `function` keep the module
    alive; calling them after `close` crashes the process."""

    def __init__(self, image: Image, memory: ExecMemory):
        self.image = image
        self.memory = memory
        self.closed = False

    def _check_open(self) -> None:
        if self.closed:
            raise LoadError("module is closed")

    def address(self, label: Label | str) -> int:
        self._check_open()
        return self.image.address(label)

    def function(self, restype: Any, *argtypes: Any, entry: Label | str | None = None) -> Any:
        """ctypes CFUNCTYPE bound to `entry` (default: image base)."""
        self._check_open()
        addr = self.image.base if entry is None else self.address(entry)
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
        fn._jita_module = self
        return fn

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
