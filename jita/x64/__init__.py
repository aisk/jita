"""x64 architecture: registers, memory operands and mnemonic functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.arch import Arch
from ..core.assembler import label
from . import insns as _insns
from . import mem as _mem
from . import regs as _regs
from .insns import *  # noqa: F403
from .insns import INSNS as _INSNS
from .mem import *  # noqa: F403
from .regs import *  # noqa: F403
from .structs import typed

# Recommended multi-byte NOP sequences (Intel SDM, NOP instruction).
_NOPS = (
    b"",
    b"\x90",
    b"\x66\x90",
    b"\x0f\x1f\x00",
    b"\x0f\x1f\x40\x00",
    b"\x0f\x1f\x44\x00\x00",
    b"\x66\x0f\x1f\x44\x00\x00",
    b"\x0f\x1f\x80\x00\x00\x00\x00",
    b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
    b"\x66\x0f\x1f\x84\x00\x00\x00\x00\x00",
)


class X64Arch(Arch):
    name = "x64"
    pointer_size = 8

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """`jita.x64.insns.INSNS`: Python name -> mnemonic function."""
        return _INSNS

    def nop_fill(self, n: int) -> bytes:
        return _NOPS[9] * (n // 9) + _NOPS[n % 9]

    def icache_flush(self, addr: int, size: int) -> None:
        pass  # x86 keeps instruction and data caches coherent

    def hole_mem(self, hole: Any, scale: int | None = None) -> _mem.MemExpr:
        """The address a gp64 register hole builds: `[hole]`, or
        `[hole*scale]` when multiplied (used by `Hole.__add__/__mul__`)."""
        if scale is None:
            return _mem.MemExpr(base=hole)
        return _mem.MemExpr(index=hole, scale=scale)


ARCH = X64Arch()

# Keep `from jita.x64 import *` to registers, size prefixes, memory operands,
# mnemonics, `typed` and ARCH; helper imports and submodule names are not
# exported.
__all__ = ["ARCH", "X64Arch", "label", "typed", *_insns.__all__, *_mem.__all__, *_regs.__all__]
