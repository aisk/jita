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

Every register is an instance of a width class: `X` (x0..x30, xzr), `W`
(w0..w30, wzr), `Sp`, `S`, `D` and `Q`. A shifted or extended register is
a `RegMod[X]` or `RegMod[W]`, so type checkers keep the width. `Cond` is
the Literal type of condition code names.
"""

from typing import TYPE_CHECKING, Any, Literal, Self

from ..core.errors import EncodeError
from ..core.operand import Operand, Register

if TYPE_CHECKING:
    from .mem import Addr

SHIFTS = ("lsl", "lsr", "asr")
EXTENDS = ("uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx")


def _amount(n: object, what: str) -> int:
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

    def lsl(self, n: int) -> RegMod[Self]:
        return RegMod(self, "lsl", _amount(n, "shift"))

    def lsr(self, n: int) -> RegMod[Self]:
        return RegMod(self, "lsr", _amount(n, "shift"))

    def asr(self, n: int) -> RegMod[Self]:
        return RegMod(self, "asr", _amount(n, "shift"))

    def __lshift__(self, n: int) -> RegMod[Self]:
        if not isinstance(n, int) or isinstance(n, bool):
            return NotImplemented
        return self.lsl(n)

    def __rshift__(self, n: int) -> RegMod[Self]:
        if not isinstance(n, int) or isinstance(n, bool):
            return NotImplemented
        return self.lsr(n)

    def _ext(self, kind: str, n: int | None) -> RegMod[Self]:
        return RegMod(self, kind, None if n is None else _amount(n, "extend"))

    def uxtb(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("uxtb", n)

    def uxth(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("uxth", n)

    def uxtw(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("uxtw", n)

    def uxtx(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("uxtx", n)

    def sxtb(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("sxtb", n)

    def sxth(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("sxth", n)

    def sxtw(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("sxtw", n)

    def sxtx(self, n: int | None = None) -> RegMod[Self]:
        return self._ext("sxtx", n)

    # address arithmetic, completed by mem[...]

    def __add__(self, other: int | Reg | RegMod[Any]) -> Addr:
        from .mem import Addr

        return Addr(self) + other

    def __radd__(self, other: int) -> Addr:
        from .mem import Addr

        return Addr(self) + other

    def __sub__(self, other: int) -> Addr:
        from .mem import Addr

        return Addr(self) - other


class _Fixed(Reg):
    # A register class with a fixed kind, size and register type.

    __slots__ = ()
    KIND = ""
    SIZE = 0
    RT = ""

    def __init__(self, name: str, code: int):
        super().__init__(name, self.KIND, code, self.SIZE, self.RT)


class X(_Fixed):
    """64 bit general purpose register: x0..x30, xzr."""

    __slots__ = ()
    KIND, SIZE, RT = "gp", 8, "x"


class W(_Fixed):
    """32 bit general purpose register: w0..w30, wzr."""

    __slots__ = ()
    KIND, SIZE, RT = "gp", 4, "w"


class Sp(_Fixed):
    """The stack pointer."""

    __slots__ = ()
    KIND, SIZE, RT = "sp", 8, "x"


class S(_Fixed):
    """Single precision FP register: s0..s31."""

    __slots__ = ()
    KIND, SIZE, RT = "fp", 4, "s"


class D(_Fixed):
    """Double precision FP register: d0..d31."""

    __slots__ = ()
    KIND, SIZE, RT = "fp", 8, "d"


class Q(_Fixed):
    """128 bit FP register: q0..q31."""

    __slots__ = ()
    KIND, SIZE, RT = "fp", 16, "q"


# Register classes for annotating helpers.
type Gp = X | W
type Fp = S | D | Q

# Condition code names accepted by b.cond, csel, cset, ccmp and friends.
type Cond = Literal[
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "hs", "lo"
]


class Mod(Operand):
    """A bare shift or extend (`lsl #16`, `uxtw`). `amount` is None when an
    extend is written without one. Instructions get one from a RegMod or
    from the `lsl=` keyword of movz/movn/movk."""

    __slots__ = ("kind", "amount")

    def __init__(self, kind: str, amount: int | None):
        if kind not in SHIFTS and kind not in EXTENDS:
            raise EncodeError(f"unknown shift or extend {kind!r}")
        if amount is not None:
            _amount(amount, "shift" if kind in SHIFTS else "extend")
        self.kind, self.amount = kind, amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mod):
            return NotImplemented
        return (self.kind, self.amount) == (other.kind, other.amount)

    def __hash__(self) -> int:
        return hash((self.kind, self.amount))

    def __str__(self) -> str:
        return self.kind if self.amount is None else f"{self.kind} #{self.amount}"

    __repr__ = __str__


class RegMod[R: Reg](Operand):
    """A register with a shift or an extend: `x1 << 3`, `w2.uxtw(2)`.

    As an instruction operand it stands for two DynASM operands, the
    register and the modifier (`x1, lsl #3`). `R` is the register class,
    `RegMod[X]` or `RegMod[W]`.
    """

    __slots__ = ("reg", "mod")

    def __init__(self, reg: R, kind: str, amount: int | None):
        self.reg, self.mod = reg, Mod(kind, amount)

    @property
    def kind(self) -> str:
        return self.mod.kind

    @property
    def amount(self) -> int | None:
        return self.mod.amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegMod):
            return NotImplemented
        return self.reg is other.reg and self.mod == other.mod

    def __hash__(self) -> int:
        return hash((id(self.reg), self.mod))

    def __str__(self) -> str:
        return f"{self.reg}, {self.mod}"

    __repr__ = __str__


x0 = X("x0", 0)
x1 = X("x1", 1)
x2 = X("x2", 2)
x3 = X("x3", 3)
x4 = X("x4", 4)
x5 = X("x5", 5)
x6 = X("x6", 6)
x7 = X("x7", 7)
x8 = X("x8", 8)
x9 = X("x9", 9)
x10 = X("x10", 10)
x11 = X("x11", 11)
x12 = X("x12", 12)
x13 = X("x13", 13)
x14 = X("x14", 14)
x15 = X("x15", 15)
x16 = X("x16", 16)
x17 = X("x17", 17)
x18 = X("x18", 18)
x19 = X("x19", 19)
x20 = X("x20", 20)
x21 = X("x21", 21)
x22 = X("x22", 22)
x23 = X("x23", 23)
x24 = X("x24", 24)
x25 = X("x25", 25)
x26 = X("x26", 26)
x27 = X("x27", 27)
x28 = X("x28", 28)
x29 = X("x29", 29)
x30 = X("x30", 30)
xzr = X("xzr", 31)

w0 = W("w0", 0)
w1 = W("w1", 1)
w2 = W("w2", 2)
w3 = W("w3", 3)
w4 = W("w4", 4)
w5 = W("w5", 5)
w6 = W("w6", 6)
w7 = W("w7", 7)
w8 = W("w8", 8)
w9 = W("w9", 9)
w10 = W("w10", 10)
w11 = W("w11", 11)
w12 = W("w12", 12)
w13 = W("w13", 13)
w14 = W("w14", 14)
w15 = W("w15", 15)
w16 = W("w16", 16)
w17 = W("w17", 17)
w18 = W("w18", 18)
w19 = W("w19", 19)
w20 = W("w20", 20)
w21 = W("w21", 21)
w22 = W("w22", 22)
w23 = W("w23", 23)
w24 = W("w24", 24)
w25 = W("w25", 25)
w26 = W("w26", 26)
w27 = W("w27", 27)
w28 = W("w28", 28)
w29 = W("w29", 29)
w30 = W("w30", 30)
wzr = W("wzr", 31)

s0 = S("s0", 0)
s1 = S("s1", 1)
s2 = S("s2", 2)
s3 = S("s3", 3)
s4 = S("s4", 4)
s5 = S("s5", 5)
s6 = S("s6", 6)
s7 = S("s7", 7)
s8 = S("s8", 8)
s9 = S("s9", 9)
s10 = S("s10", 10)
s11 = S("s11", 11)
s12 = S("s12", 12)
s13 = S("s13", 13)
s14 = S("s14", 14)
s15 = S("s15", 15)
s16 = S("s16", 16)
s17 = S("s17", 17)
s18 = S("s18", 18)
s19 = S("s19", 19)
s20 = S("s20", 20)
s21 = S("s21", 21)
s22 = S("s22", 22)
s23 = S("s23", 23)
s24 = S("s24", 24)
s25 = S("s25", 25)
s26 = S("s26", 26)
s27 = S("s27", 27)
s28 = S("s28", 28)
s29 = S("s29", 29)
s30 = S("s30", 30)
s31 = S("s31", 31)

d0 = D("d0", 0)
d1 = D("d1", 1)
d2 = D("d2", 2)
d3 = D("d3", 3)
d4 = D("d4", 4)
d5 = D("d5", 5)
d6 = D("d6", 6)
d7 = D("d7", 7)
d8 = D("d8", 8)
d9 = D("d9", 9)
d10 = D("d10", 10)
d11 = D("d11", 11)
d12 = D("d12", 12)
d13 = D("d13", 13)
d14 = D("d14", 14)
d15 = D("d15", 15)
d16 = D("d16", 16)
d17 = D("d17", 17)
d18 = D("d18", 18)
d19 = D("d19", 19)
d20 = D("d20", 20)
d21 = D("d21", 21)
d22 = D("d22", 22)
d23 = D("d23", 23)
d24 = D("d24", 24)
d25 = D("d25", 25)
d26 = D("d26", 26)
d27 = D("d27", 27)
d28 = D("d28", 28)
d29 = D("d29", 29)
d30 = D("d30", 30)
d31 = D("d31", 31)

q0 = Q("q0", 0)
q1 = Q("q1", 1)
q2 = Q("q2", 2)
q3 = Q("q3", 3)
q4 = Q("q4", 4)
q5 = Q("q5", 5)
q6 = Q("q6", 6)
q7 = Q("q7", 7)
q8 = Q("q8", 8)
q9 = Q("q9", 9)
q10 = Q("q10", 10)
q11 = Q("q11", 11)
q12 = Q("q12", 12)
q13 = Q("q13", 13)
q14 = Q("q14", 14)
q15 = Q("q15", 15)
q16 = Q("q16", 16)
q17 = Q("q17", 17)
q18 = Q("q18", 18)
q19 = Q("q19", 19)
q20 = Q("q20", 20)
q21 = Q("q21", 21)
q22 = Q("q22", 22)
q23 = Q("q23", 23)
q24 = Q("q24", 24)
q25 = Q("q25", 25)
q26 = Q("q26", 26)
q27 = Q("q27", 27)
q28 = Q("q28", 28)
q29 = Q("q29", 29)
q30 = Q("q30", 30)
q31 = Q("q31", 31)

# The stack pointer shares number 31 with xzr; the operand position decides
# which of the two an instruction means.
sp = Sp("sp", 31)

# Aliases: the same objects under another name.
lr = x30
fp = x29

# Register number -> register, for gp64(n) and friends.
_x = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, xzr)
_w = (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, wzr)
_s = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31)
_d = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31)
_q = (q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, q24, q25, q26, q27, q28, q29, q30, q31)


def _pick[R: Reg](table: tuple[R, ...], n: int, what: str) -> R:
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < len(table):
        raise EncodeError(f"{what}: register number {n!r} out of range")
    return table[n]


def gp64(n: int) -> X:
    """Return x<n> (n in 0..31, 31 is xzr)."""
    return _pick(_x, n, "gp64")


def gp32(n: int) -> W:
    """Return w<n> (n in 0..31, 31 is wzr)."""
    return _pick(_w, n, "gp32")


def fp32(n: int) -> S:
    """Return the single precision register s<n> (n in 0..31)."""
    return _pick(_s, n, "fp32")


def fp64(n: int) -> D:
    """Return the double precision register d<n> (n in 0..31)."""
    return _pick(_d, n, "fp64")


def fp128(n: int) -> Q:
    """Return the 128 bit register q<n> (n in 0..31)."""
    return _pick(_q, n, "fp128")


ALL_REGS: dict[str, Reg] = {r.name: r for r in (*_x, *_w, *_s, *_d, *_q, sp)}

__all__ = [
    "Reg", "X", "W", "Sp", "S", "D", "Q", "Gp", "Fp", "Cond", "RegMod", "Mod", "gp32",
    "gp64", "fp32", "fp64", "fp128", "lr", "fp", "x0", "x1", "x2", "x3", "x4", "x5",
    "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
    "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
    "x30", "xzr", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10",
    "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22",
    "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wzr", "s0", "s1", "s2",
    "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
    "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
    "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "q0",
    "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13",
    "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25",
    "q26", "q27", "q28", "q29", "q30", "q31", "sp",
]
