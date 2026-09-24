"""x64 register objects."""

from __future__ import annotations

from ..core.errors import EncodeError
from ..core.operand import Register


class Reg(Register):
    """An x64 register.

    `kind` is one of "gp", "xmm", "ymm", "st" (x87 stack), "rip". `high` marks ah/ch/dh/bh,
    which cannot be encoded together with a REX prefix. Arithmetic on
    registers builds memory operands: `rbx + rcx*8 + 16`.
    """

    __slots__ = ("high",)

    def __init__(self, name: str, kind: str, code: int, size: int, high: bool = False):
        super().__init__(name, kind, code, size)
        self.high = high

    @property
    def needs_rex(self) -> bool:
        """True if any encoding of this register requires a REX prefix."""
        return self.code >= 8 or (self.kind == "gp" and self.size == 1 and self.code >= 4 and not self.high)

    def __str__(self) -> str:
        # GNU as spells x87 stack registers st(0)..st(7).
        return f"st({self.code})" if self.kind == "st" else self.name

    def _mem(self):
        from .mem import MemExpr

        return MemExpr(base=self)

    def __add__(self, other):
        return self._mem() + other

    __radd__ = __add__

    def __sub__(self, other):
        return self._mem() - other

    def __mul__(self, scale):
        from .mem import MemExpr

        if not isinstance(scale, int) or isinstance(scale, bool):
            return NotImplemented
        return MemExpr(index=self, scale=scale)

    __rmul__ = __mul__


_GP64 = "rax rcx rdx rbx rsp rbp rsi rdi".split()
_GP32 = "eax ecx edx ebx esp ebp esi edi".split()
_GP16 = "ax cx dx bx sp bp si di".split()
_GP8 = "al cl dl bl spl bpl sil dil".split()

_gp8 = tuple(Reg(n, "gp", i, 1) for i, n in enumerate(_GP8 + [f"r{i}b" for i in range(8, 16)]))
_gp16 = tuple(Reg(n, "gp", i, 2) for i, n in enumerate(_GP16 + [f"r{i}w" for i in range(8, 16)]))
_gp32 = tuple(Reg(n, "gp", i, 4) for i, n in enumerate(_GP32 + [f"r{i}d" for i in range(8, 16)]))
_gp64 = tuple(Reg(n, "gp", i, 8) for i, n in enumerate(_GP64 + [f"r{i}" for i in range(8, 16)]))
_xmm = tuple(Reg(f"xmm{i}", "xmm", i, 16) for i in range(16))
_ymm = tuple(Reg(f"ymm{i}", "ymm", i, 32) for i in range(16))
_st = tuple(Reg(f"st{i}", "st", i, 10) for i in range(8))
_high = tuple(Reg(n, "gp", 4 + i, 1, high=True) for i, n in enumerate(["ah", "ch", "dh", "bh"]))

# rip is only valid as a memory base. Its code is the ModRM r/m value (101)
# that selects rip-relative addressing with mod=00.
rip = Reg("rip", "rip", 5, 8)


def _pick(table: tuple[Reg, ...], what: str):
    def ctor(n: int) -> Reg:
        if not isinstance(n, int) or not 0 <= n < len(table):
            raise EncodeError(f"{what}: register number {n!r} out of range")
        return table[n]

    ctor.__name__ = what
    ctor.__doc__ = f"Return the {what} register with hardware number n (0..15)."
    return ctor


gp8 = _pick(_gp8, "gp8")
gp16 = _pick(_gp16, "gp16")
gp32 = _pick(_gp32, "gp32")
gp64 = _pick(_gp64, "gp64")
xmm = _pick(_xmm, "xmm")
ymm = _pick(_ymm, "ymm")
st = _pick(_st, "st")
st.__doc__ = "Return the x87 stack register st(n) (n in 0..7)."

ALL_REGS: dict[str, Reg] = {r.name: r for r in (*_gp8, *_high, *_gp16, *_gp32, *_gp64, *_xmm, *_ymm, *_st, rip)}
globals().update(ALL_REGS)

__all__ = ["Reg", "gp8", "gp16", "gp32", "gp64", "xmm", "ymm", "st", *ALL_REGS]
