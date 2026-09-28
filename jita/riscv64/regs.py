"""riscv64 register objects.

    zero ra sp gp tp t0-t6 s0-s11 a0-a7   integer registers (class `X`)
    ft0-ft11 fs0-fs11 fa0-fa7             floating point registers (class `F`)
    x0..x31, f0..f31                      the same objects by number
    fp                                    alias of s0 (x8)

Registers are named by their ABI name, as objdump prints them; `x10` and
`a0` are one object whose name is "a0". There are no width classes: the
mnemonic decides the access size (`lw`, `ld`) and the precision
(`fadd.s`, `fadd.d`).

`sp + 8` and `a0 - 16` build addresses for `mem[...]` (see
`jita.riscv64.mem`).

`Rm` is the Literal type of rounding mode names (the `rm=` keyword of
floating point instructions), `CsrName` the CSR names accepted in place of
a number and `FenceSet` the predecessor and successor sets of `fence`.
"""

from typing import TYPE_CHECKING, Literal

from ..core.errors import EncodeError
from ..core.operand import Register

if TYPE_CHECKING:
    from .mem import Addr


class Reg(Register):
    """A riscv64 register. `kind` is "gp" or "fp"."""

    __slots__ = ()

    # address arithmetic, completed by mem[...]

    def __add__(self, other: int) -> Addr:
        from .mem import Addr

        return Addr(self) + other

    def __radd__(self, other: int) -> Addr:
        from .mem import Addr

        return Addr(self) + other

    def __sub__(self, other: int) -> Addr:
        from .mem import Addr

        return Addr(self) - other


class X(Reg):
    """Integer register: x0..x31 (zero, ra, sp, ..., t6)."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "gp", code, 8)


class F(Reg):
    """Floating point register: f0..f31 (ft0, ..., ft11)."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "fp", code, 8)


# Rounding modes of the `rm=` keyword.
type Rm = Literal["rne", "rtz", "rdn", "rup", "rmm", "dyn"]

# CSR names accepted where a CSR number is.
type CsrName = Literal["fflags", "frm", "fcsr", "cycle", "time", "instret"]

# fence predecessor and successor sets, in the order GNU as requires.
type FenceSet = Literal[
    "i", "o", "r", "w", "io", "ir", "iw", "or", "ow", "rw", "ior", "iow", "irw", "orw", "iorw"
]


zero = X("zero", 0)
ra = X("ra", 1)
sp = X("sp", 2)
gp = X("gp", 3)
tp = X("tp", 4)
t0 = X("t0", 5)
t1 = X("t1", 6)
t2 = X("t2", 7)
s0 = X("s0", 8)
s1 = X("s1", 9)
a0 = X("a0", 10)
a1 = X("a1", 11)
a2 = X("a2", 12)
a3 = X("a3", 13)
a4 = X("a4", 14)
a5 = X("a5", 15)
a6 = X("a6", 16)
a7 = X("a7", 17)
s2 = X("s2", 18)
s3 = X("s3", 19)
s4 = X("s4", 20)
s5 = X("s5", 21)
s6 = X("s6", 22)
s7 = X("s7", 23)
s8 = X("s8", 24)
s9 = X("s9", 25)
s10 = X("s10", 26)
s11 = X("s11", 27)
t3 = X("t3", 28)
t4 = X("t4", 29)
t5 = X("t5", 30)
t6 = X("t6", 31)

ft0 = F("ft0", 0)
ft1 = F("ft1", 1)
ft2 = F("ft2", 2)
ft3 = F("ft3", 3)
ft4 = F("ft4", 4)
ft5 = F("ft5", 5)
ft6 = F("ft6", 6)
ft7 = F("ft7", 7)
fs0 = F("fs0", 8)
fs1 = F("fs1", 9)
fa0 = F("fa0", 10)
fa1 = F("fa1", 11)
fa2 = F("fa2", 12)
fa3 = F("fa3", 13)
fa4 = F("fa4", 14)
fa5 = F("fa5", 15)
fa6 = F("fa6", 16)
fa7 = F("fa7", 17)
fs2 = F("fs2", 18)
fs3 = F("fs3", 19)
fs4 = F("fs4", 20)
fs5 = F("fs5", 21)
fs6 = F("fs6", 22)
fs7 = F("fs7", 23)
fs8 = F("fs8", 24)
fs9 = F("fs9", 25)
fs10 = F("fs10", 26)
fs11 = F("fs11", 27)
ft8 = F("ft8", 28)
ft9 = F("ft9", 29)
ft10 = F("ft10", 30)
ft11 = F("ft11", 31)

# Numeric and alias names: the same objects under another name.
x0 = zero
x1 = ra
x2 = sp
x3 = gp
x4 = tp
x5 = t0
x6 = t1
x7 = t2
x8 = s0
x9 = s1
x10 = a0
x11 = a1
x12 = a2
x13 = a3
x14 = a4
x15 = a5
x16 = a6
x17 = a7
x18 = s2
x19 = s3
x20 = s4
x21 = s5
x22 = s6
x23 = s7
x24 = s8
x25 = s9
x26 = s10
x27 = s11
x28 = t3
x29 = t4
x30 = t5
x31 = t6

f0 = ft0
f1 = ft1
f2 = ft2
f3 = ft3
f4 = ft4
f5 = ft5
f6 = ft6
f7 = ft7
f8 = fs0
f9 = fs1
f10 = fa0
f11 = fa1
f12 = fa2
f13 = fa3
f14 = fa4
f15 = fa5
f16 = fa6
f17 = fa7
f18 = fs2
f19 = fs3
f20 = fs4
f21 = fs5
f22 = fs6
f23 = fs7
f24 = fs8
f25 = fs9
f26 = fs10
f27 = fs11
f28 = ft8
f29 = ft9
f30 = ft10
f31 = ft11
fp = s0

# Register number -> register, for gpr(n) and fpr(n).
_x = (zero, ra, sp, gp, tp, t0, t1, t2, s0, s1, a0, a1, a2, a3, a4, a5, a6, a7, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, t3, t4, t5, t6)
_f = (ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, fs0, fs1, fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7, fs2, fs3, fs4, fs5, fs6, fs7, fs8, fs9, fs10, fs11, ft8, ft9, ft10, ft11)


def _pick[R: Reg](table: tuple[R, ...], n: int, what: str) -> R:
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < len(table):
        raise EncodeError(f"{what}: register number {n!r} out of range")
    return table[n]


def gpr(n: int) -> X:
    """Return the integer register x<n> (n in 0..31, 0 is zero)."""
    return _pick(_x, n, "gpr")


def fpr(n: int) -> F:
    """Return the floating point register f<n> (n in 0..31)."""
    return _pick(_f, n, "fpr")


ALL_REGS: dict[str, Reg] = {r.name: r for r in (*_x, *_f)}

__all__ = [
    "Reg", "X", "F", "Rm", "CsrName", "FenceSet", "gpr", "fpr", "fp", "zero", "ra",
    "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3",
    "t4", "t5", "t6", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0",
    "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4",
    "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
    "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24",
    "x25", "x26", "x27", "x28", "x29", "x30", "x31", "f0", "f1", "f2", "f3", "f4", "f5",
    "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17",
    "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29",
    "f30", "f31",
]
