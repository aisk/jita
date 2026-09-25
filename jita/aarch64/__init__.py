"""aarch64 architecture: registers, memory operands and mnemonic functions."""

from __future__ import annotations

import ctypes
import platform
from collections.abc import Callable
from typing import Any

from ..core.arch import Arch
from ..core.assembler import label
from ..core.errors import EncodeError, LoadError
from . import insns as _insns
from . import mem as _mem
from . import regs as _regs
from .insns import *  # noqa: F403
from .insns import INSNS as _INSNS
from .mem import *  # noqa: F403
from .regs import *  # noqa: F403

NOP = b"\x1f\x20\x03\xd5"  # d503201f


def _host_is_aarch64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


class Aarch64Arch(Arch):
    name = "aarch64"
    pointer_size = 8

    def __init__(self) -> None:
        self._clear_cache: Callable[..., Any] | None = None

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """`jita.aarch64.insns.INSNS`: Python name -> mnemonic function."""
        return _INSNS

    def nop_fill(self, n: int) -> bytes:
        """n bytes of padding: zero bytes up to the next multiple of 4, then
        NOP words, so the NOPs stay word aligned when the padding starts
        at an instruction boundary."""
        return bytes(n % 4) + NOP * (n // 4)

    def hole_mem(self, hole: Any, scale: int | None = None) -> _mem.Addr:
        """The address a gp64 register hole starts: `hole + 8` is the same
        `Addr` as `x0 + 8`, completed by `mem[...]` (used by
        `Hole.__add__`). aarch64 has no `index*scale` addresses; a scaled
        index is written `mem[base + (index << n)]`."""
        if scale is not None:
            raise EncodeError("aarch64 addresses have no index*scale, write mem[base + (index << n)]")
        if not (hole.regclass == "gp" and hole.size == 8):
            raise EncodeError(f"{hole!r} cannot be a memory base or index (use a gp64 hole)")
        return _mem.Addr(hole)

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

# Keep `from jita.aarch64 import *` to registers, memory operands, mnemonics
# and ARCH; helper imports and submodule names are not exported.
__all__ = ["ARCH", "Aarch64Arch", "label", *_insns.__all__, *_mem.__all__, *_regs.__all__]
