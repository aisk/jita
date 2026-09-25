"""aarch64 register objects and register modifiers.

    x0..x30, xzr          64 bit general purpose registers (kind "gp", size 8)
    w0..w30, wzr          32 bit views (kind "gp", size 4)
    sp                    stack pointer (kind "sp"), accepted as a memory
                          base and in the operands that encode register 31
                          as the stack pointer
    s0..s31, d0..d31      single and double precision FP registers (kind "fp")
    q0..q31               128 bit FP registers (ldp/stp only)
    lr, fp                aliases of x30 and x29

A register with a shift or an extend is written as an expression:

    x1 << 3, x1.lsl(3)    x1, lsl #3
    x1 >> 3, x1.lsr(3)    x1, lsr #3
    x1.asr(3)             x1, asr #3
    w1.uxtw(2), w1.sxtw() w1, uxtw #2 / w1, sxtw (also uxtb uxth uxtx
                          sxtb sxth sxtx; the amount is optional)

`x0 + 8`, `x0 + x1` and `x0 + (x1 << 3)` build addresses for `mem[...]`
(see `jita.aarch64.mem`).
"""

from __future__ import annotations

from ..core.errors import EncodeError
from ..core.operand import Operand, Register

SHIFTS = ("lsl", "lsr", "asr")
EXTENDS = ("uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx")


def _amount(n, what: str) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise EncodeError(f"{what} amount must be a non-negative int, got {n!r}")
    return n


class Reg(Register):
    """An aarch64 register.

    `rt` is DynASM's register type letter: "x" or "w" for general purpose
    registers, "s", "d" or "q" for FP registers. `sp` has kind "sp" and
    rt "x".
    """

    __slots__ = ("rt",)

    def __init__(self, name: str, kind: str, code: int, size: int, rt: str):
        super().__init__(name, kind, code, size)
        self.rt = rt

    # shifted and extended register operands

    def lsl(self, n: int) -> RegMod:
        return RegMod(self, "lsl", _amount(n, "shift"))

    def lsr(self, n: int) -> RegMod:
        return RegMod(self, "lsr", _amount(n, "shift"))

    def asr(self, n: int) -> RegMod:
        return RegMod(self, "asr", _amount(n, "shift"))

    def __lshift__(self, n):
        if not isinstance(n, int) or isinstance(n, bool):
            return NotImplemented
        return self.lsl(n)

    def __rshift__(self, n):
        if not isinstance(n, int) or isinstance(n, bool):
            return NotImplemented
        return self.lsr(n)

    def _ext(self, kind: str, n: int | None) -> RegMod:
        return RegMod(self, kind, None if n is None else _amount(n, "extend"))

    def uxtb(self, n: int | None = None) -> RegMod:
        return self._ext("uxtb", n)

    def uxth(self, n: int | None = None) -> RegMod:
        return self._ext("uxth", n)

    def uxtw(self, n: int | None = None) -> RegMod:
        return self._ext("uxtw", n)

    def uxtx(self, n: int | None = None) -> RegMod:
        return self._ext("uxtx", n)

    def sxtb(self, n: int | None = None) -> RegMod:
        return self._ext("sxtb", n)

    def sxth(self, n: int | None = None) -> RegMod:
        return self._ext("sxth", n)

    def sxtw(self, n: int | None = None) -> RegMod:
        return self._ext("sxtw", n)

    def sxtx(self, n: int | None = None) -> RegMod:
        return self._ext("sxtx", n)

    # address arithmetic, completed by mem[...]

    def __add__(self, other):
        from .mem import Addr

        return Addr(self) + other

    def __sub__(self, other):
        from .mem import Addr

        return Addr(self) - other


class Mod(Operand):
    """A bare shift or extend (`lsl #16`, `uxtw`). `amount` is None when an
    extend is written without one. Instructions get one from a RegMod or
    from the `lsl=` keyword of movz/movn/movk."""

    __slots__ = ("kind", "amount")

    def __init__(self, kind: str, amount: int | None):
        if kind not in SHIFTS and kind not in EXTENDS:
            raise EncodeError(f"unknown shift or extend {kind!r}")
        self.kind, self.amount = kind, amount

    def __eq__(self, other):
        if not isinstance(other, Mod):
            return NotImplemented
        return (self.kind, self.amount) == (other.kind, other.amount)

    def __hash__(self):
        return hash((self.kind, self.amount))

    def __str__(self) -> str:
        return self.kind if self.amount is None else f"{self.kind} #{self.amount}"

    __repr__ = __str__


class RegMod(Operand):
    """A register with a shift or an extend: `x1 << 3`, `w2.uxtw(2)`.

    As an instruction operand it stands for two DynASM operands, the
    register and the modifier (`x1, lsl #3`).
    """

    __slots__ = ("reg", "mod")

    def __init__(self, reg: Reg, kind: str, amount: int | None):
        self.reg, self.mod = reg, Mod(kind, amount)

    @property
    def kind(self) -> str:
        return self.mod.kind

    @property
    def amount(self) -> int | None:
        return self.mod.amount

    def __eq__(self, other):
        if not isinstance(other, RegMod):
            return NotImplemented
        return self.reg is other.reg and self.mod == other.mod

    def __hash__(self):
        return hash((id(self.reg), self.mod))

    def __str__(self) -> str:
        return f"{self.reg}, {self.mod}"

    __repr__ = __str__


_x = tuple(Reg(f"x{i}", "gp", i, 8, "x") for i in range(31)) + (Reg("xzr", "gp", 31, 8, "x"),)
_w = tuple(Reg(f"w{i}", "gp", i, 4, "w") for i in range(31)) + (Reg("wzr", "gp", 31, 4, "w"),)
_s = tuple(Reg(f"s{i}", "fp", i, 4, "s") for i in range(32))
_d = tuple(Reg(f"d{i}", "fp", i, 8, "d") for i in range(32))
_q = tuple(Reg(f"q{i}", "fp", i, 16, "q") for i in range(32))

# The stack pointer shares number 31 with xzr; the operand position decides
# which of the two an instruction means.
sp = Reg("sp", "sp", 31, 8, "x")


def _pick(table: tuple[Reg, ...], what: str, doc: str):
    def ctor(n: int) -> Reg:
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < len(table):
            raise EncodeError(f"{what}: register number {n!r} out of range")
        return table[n]

    ctor.__name__ = ctor.__qualname__ = what
    ctor.__doc__ = doc
    return ctor


gp64 = _pick(_x, "gp64", "Return x<n> (n in 0..31, 31 is xzr).")
gp32 = _pick(_w, "gp32", "Return w<n> (n in 0..31, 31 is wzr).")
fp32 = _pick(_s, "fp32", "Return the single precision register s<n> (n in 0..31).")
fp64 = _pick(_d, "fp64", "Return the double precision register d<n> (n in 0..31).")
fp128 = _pick(_q, "fp128", "Return the 128 bit register q<n> (n in 0..31).")

ALL_REGS: dict[str, Reg] = {r.name: r for r in (*_x, *_w, *_s, *_d, *_q, sp)}
globals().update(ALL_REGS)

# Aliases: the same objects under another name.
lr = _x[30]
fp = _x[29]

__all__ = ["Reg", "RegMod", "Mod", "gp32", "gp64", "fp32", "fp64", "fp128", "lr", "fp", *ALL_REGS]
