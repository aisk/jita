"""aarch64 architecture: registers, memory operands and mnemonic functions."""

# __all__ is computed; the stub next to this module carries it literally.
# pyright: reportUnsupportedDunderAll=false

import ctypes
import platform
from collections.abc import Callable
from typing import Any

from ..core.arch import Arch
from ..core.assembler import label
from ..core.errors import LoadError
from . import insns as _insns
from . import mem as _mem
from . import regs as _regs
from .insns import *  # noqa: F403
from .insns import INSNS as _INSNS
from .insns import Aarch64Assembler
from .mem import *  # noqa: F403
from .regs import *  # noqa: F403

NOP = b"\x1f\x20\x03\xd5"  # d503201f


def _host_is_aarch64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


class Aarch64Arch(Arch):
    name = "aarch64"
    pointer_size = 8
    imm_prefix = "#"

    def __init__(self) -> None:
        self._clear_cache: Callable[..., Any] | None = None

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """`jita.aarch64.insns.INSNS`: Python name -> mnemonic function."""
        return _INSNS

    @property
    def assembler_class(self) -> type[Aarch64Assembler]:
        return Aarch64Assembler

    def nop_fill(self, n: int) -> bytes:
        """n bytes of padding: zero bytes up to the next multiple of 4, then
        NOP words, so the NOPs stay word aligned when the padding starts
        at an instruction boundary."""
        return bytes(n % 4) + NOP * (n // 4)

    def icache_flush(self, addr: int, size: int) -> None:
        """Call libgcc's `__clear_cache` on an aarch64 host; no-op elsewhere
        (code for another architecture is never executed there)."""
        if size <= 0 or not _host_is_aarch64():
            return
        fn = self._clear_cache
        if fn is None:
            fn = self._clear_cache = _find_clear_cache()
        fn(ctypes.c_void_p(addr), ctypes.c_void_p(addr + size))


def _find_clear_cache() -> Callable[..., Any]:
    for lib in (None, "libgcc_s.so.1", "libgcc_s.1.dylib", "libSystem.B.dylib"):
        try:
            fn = getattr(ctypes.CDLL(lib), "__clear_cache")
        except (OSError, AttributeError):
            continue
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return fn
    raise LoadError("__clear_cache not found (libgcc_s), cannot flush the instruction cache")


ARCH = Aarch64Arch()

# Keep `from jita.aarch64 import *` to registers, memory operands, mnemonics,
# ARCH and the arch classes; helper imports and submodule names are not
# exported. `__init__.pyi` carries the same list literally.
__all__ = [
    "ARCH",
    "Aarch64Arch",
    "Aarch64Assembler",
    "label",
    *_insns.__all__,
    *_mem.__all__,
    *_regs.__all__,
]
