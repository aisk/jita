"""riscv64 architecture: registers, memory operands and mnemonic functions."""

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
from .insns import Riscv64Assembler
from .mem import *  # noqa: F403
from .regs import *  # noqa: F403

NOP = b"\x13\x00\x00\x00"  # 00000013, addi zero, zero, 0

# __NR_riscv_flush_icache in the Linux generic syscall table.
SYS_RISCV_FLUSH_ICACHE = 259


def _host_is_riscv64() -> bool:
    return platform.machine().lower() == "riscv64"


class Riscv64Arch(Arch):
    name = "riscv64"
    pointer_size = 8
    imm_prefix = ""

    def __init__(self) -> None:
        self._flush: Callable[[int, int], None] | None = None

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """`jita.riscv64.insns.INSNS`: Python name -> mnemonic function."""
        return _INSNS

    @property
    def assembler_class(self) -> type[Riscv64Assembler]:
        return Riscv64Assembler

    def nop_fill(self, n: int) -> bytes:
        """n bytes of padding: zero bytes up to the next multiple of 4, then
        NOP words, so the NOPs stay word aligned when the padding starts
        at an instruction boundary."""
        return bytes(n % 4) + NOP * (n // 4)

    def icache_flush(self, addr: int, size: int) -> None:
        """Call libgcc's `__clear_cache` on a riscv64 host, or the
        riscv_flush_icache system call when libgcc is not found; no-op
        elsewhere (code for another architecture is never executed there)."""
        if size <= 0 or not _host_is_riscv64():
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
    return _syscall_flush()


def _syscall_flush() -> Callable[[int, int], None]:
    """riscv_flush_icache(start, end, 0) through libc's `syscall`, for
    hosts without libgcc_s."""
    try:
        syscall = ctypes.CDLL(None, use_errno=True).syscall
    except (OSError, AttributeError):
        raise LoadError("neither __clear_cache nor syscall found, cannot flush the instruction cache") from None
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]

    def flush(start: int, end: int) -> None:
        # flags 0: make the range coherent on all harts, not just this one.
        if syscall(SYS_RISCV_FLUSH_ICACHE, ctypes.c_void_p(start), ctypes.c_void_p(end), 0) != 0:
            raise LoadError(f"riscv_flush_icache failed: errno {ctypes.get_errno()}")

    return flush


ARCH = Riscv64Arch()

# Keep `from jita.riscv64 import *` to registers, memory operands, mnemonics,
# ARCH and the arch classes; helper imports and submodule names are not
# exported. `__init__.pyi` carries the same list literally.
__all__ = [
    "ARCH",
    "Riscv64Arch",
    "Riscv64Assembler",
    "label",
    *_insns.__all__,
    *_mem.__all__,
    *_regs.__all__,
]
