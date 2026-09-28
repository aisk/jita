"""loongarch64 architecture: registers and mnemonic functions."""

# __all__ is computed; the stub next to this module carries it literally.
# pyright: reportUnsupportedDunderAll=false

import ctypes
import platform
from collections.abc import Callable
from typing import Any

from ..core.arch import Arch
from ..core.assembler import label
from . import insns as _insns
from . import regs as _regs
from .insns import *  # noqa: F403
from .insns import INSNS as _INSNS
from .insns import Loongarch64Assembler
from .regs import *  # noqa: F403

NOP = b"\x00\x00\x40\x03"  # 03400000, andi zero, zero, 0

# `ibar 0; ret`: what libgcc's __clear_cache is on loongarch64.
IBAR_RET = b"\x00\x80\x72\x38" + b"\x20\x00\x00\x4c"


def _host_is_loongarch64() -> bool:
    return platform.machine().lower() == "loongarch64"


class Loongarch64Arch(Arch):
    name = "loongarch64"
    pointer_size = 8
    imm_prefix = ""

    def __init__(self) -> None:
        self._flush: Callable[[int, int], None] | None = None

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """`jita.loongarch64.insns.INSNS`: Python name -> mnemonic function."""
        return _INSNS

    @property
    def assembler_class(self) -> type[Loongarch64Assembler]:
        return Loongarch64Assembler

    def nop_fill(self, n: int) -> bytes:
        """n bytes of padding: zero bytes up to the next multiple of 4, then
        NOP words, so the NOPs stay word aligned when the padding starts
        at an instruction boundary."""
        return bytes(n % 4) + NOP * (n // 4)

    def icache_flush(self, addr: int, size: int) -> None:
        """Call libgcc's `__clear_cache` (an `ibar 0`) on a loongarch64
        host, or a JIT compiled `ibar 0` when libgcc is not found; no-op
        elsewhere (code for another architecture is never executed
        there)."""
        if size <= 0 or not _host_is_loongarch64():
            return
        fn = self._flush
        if fn is None:
            fn = self._flush = _find_flush()
        fn(addr, addr + size)


def _find_flush() -> Callable[[int, int], None]:
    for lib in (None, "libgcc_s.so.1"):
        try:
            clear = getattr(ctypes.CDLL(lib), "__clear_cache")
        except (OSError, AttributeError):
            continue
        clear.restype = None
        clear.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return lambda start, end: clear(ctypes.c_void_p(start), ctypes.c_void_p(end))
    return _stub_flush()


def _stub_flush() -> Callable[[int, int], None]:
    """An `ibar 0; ret` function of our own, for hosts without libgcc_s.
    `ibar 0` orders all earlier stores before later instruction fetches
    on this hart, whatever the range, so the arguments are not needed.
    The stub itself is new code too: it is written before mprotect makes
    it executable, and the system call's return already orders that."""
    from ..runtime.memory import ExecMemory

    mem = ExecMemory(len(IBAR_RET))
    mem.write(IBAR_RET)
    mem.protect_exec()
    stub = ctypes.CFUNCTYPE(None)(mem.address)

    def flush(start: int, end: int) -> None:
        stub()

    # Keep the mapping alive as long as the function.
    setattr(flush, "memory", mem)
    return flush


ARCH = Loongarch64Arch()

# Keep `from jita.loongarch64 import *` to registers, mnemonics, ARCH and
# the arch classes; helper imports and submodule names are not exported.
# `__init__.pyi` carries the same list literally.
__all__ = [
    "ARCH",
    "Loongarch64Arch",
    "Loongarch64Assembler",
    "label",
    *_insns.__all__,
    *_regs.__all__,
]
