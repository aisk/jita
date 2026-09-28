"""loongarch64 register objects.

    zero ra tp sp a0-a7 t0-t8 r21 fp s0-s8     integer registers (class `R`)
    fa0-fa7 ft0-ft15 fs0-fs7                   floating point registers (class `F`)
    fcc0..fcc7                                 condition flags (class `Fcc`)
    fcsr0..fcsr3                               FP control and status (class `Fcsr`)
    r0..r31, f0..f31                           the same objects by number
    s9                                         alias of fp (r22)

Registers are named by their ABI name, as objdump prints them; `r4` and
`a0` are one object whose name is "a0". `str()` gives the GNU spelling
with a `$` (`$a0`), which listings print. r21 has no ABI name in GNU as
and is `r21`. There are no width classes: the mnemonic decides the
access size and the precision (`ld.w`, `ld.d`, `fadd.s`, `fadd.d`).
"""

from ..core.errors import EncodeError
from ..core.operand import Register


class Reg(Register):
    """A loongarch64 register. `kind` is "gp", "fp", "fcc" or "fcsr"."""

    __slots__ = ()

    def __str__(self) -> str:
        return "$" + self.name


class R(Reg):
    """Integer register: r0..r31 (zero, ra, tp, sp, ..., s8)."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "gp", code, 8)


class F(Reg):
    """Floating point register: f0..f31 (fa0, ..., fs7)."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "fp", code, 8)


class Fcc(Reg):
    """Floating point condition flag register: fcc0..fcc7."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "fcc", code, 1)


class Fcsr(Reg):
    """Floating point control and status register: fcsr0..fcsr3."""

    __slots__ = ()

    def __init__(self, name: str, code: int):
        super().__init__(name, "fcsr", code, 4)


zero = R("zero", 0)
ra = R("ra", 1)
tp = R("tp", 2)
sp = R("sp", 3)
a0 = R("a0", 4)
a1 = R("a1", 5)
a2 = R("a2", 6)
a3 = R("a3", 7)
a4 = R("a4", 8)
a5 = R("a5", 9)
a6 = R("a6", 10)
a7 = R("a7", 11)
t0 = R("t0", 12)
t1 = R("t1", 13)
t2 = R("t2", 14)
t3 = R("t3", 15)
t4 = R("t4", 16)
t5 = R("t5", 17)
t6 = R("t6", 18)
t7 = R("t7", 19)
t8 = R("t8", 20)
r21 = R("r21", 21)
fp = R("fp", 22)
s0 = R("s0", 23)
s1 = R("s1", 24)
s2 = R("s2", 25)
s3 = R("s3", 26)
s4 = R("s4", 27)
s5 = R("s5", 28)
s6 = R("s6", 29)
s7 = R("s7", 30)
s8 = R("s8", 31)

fa0 = F("fa0", 0)
fa1 = F("fa1", 1)
fa2 = F("fa2", 2)
fa3 = F("fa3", 3)
fa4 = F("fa4", 4)
fa5 = F("fa5", 5)
fa6 = F("fa6", 6)
fa7 = F("fa7", 7)
ft0 = F("ft0", 8)
ft1 = F("ft1", 9)
ft2 = F("ft2", 10)
ft3 = F("ft3", 11)
ft4 = F("ft4", 12)
ft5 = F("ft5", 13)
ft6 = F("ft6", 14)
ft7 = F("ft7", 15)
ft8 = F("ft8", 16)
ft9 = F("ft9", 17)
ft10 = F("ft10", 18)
ft11 = F("ft11", 19)
ft12 = F("ft12", 20)
ft13 = F("ft13", 21)
ft14 = F("ft14", 22)
ft15 = F("ft15", 23)
fs0 = F("fs0", 24)
fs1 = F("fs1", 25)
fs2 = F("fs2", 26)
fs3 = F("fs3", 27)
fs4 = F("fs4", 28)
fs5 = F("fs5", 29)
fs6 = F("fs6", 30)
fs7 = F("fs7", 31)

fcc0 = Fcc("fcc0", 0)
fcc1 = Fcc("fcc1", 1)
fcc2 = Fcc("fcc2", 2)
fcc3 = Fcc("fcc3", 3)
fcc4 = Fcc("fcc4", 4)
fcc5 = Fcc("fcc5", 5)
fcc6 = Fcc("fcc6", 6)
fcc7 = Fcc("fcc7", 7)

fcsr0 = Fcsr("fcsr0", 0)
fcsr1 = Fcsr("fcsr1", 1)
fcsr2 = Fcsr("fcsr2", 2)
fcsr3 = Fcsr("fcsr3", 3)

# Numeric and alias names: the same objects under another name.
r0 = zero
r1 = ra
r2 = tp
r3 = sp
r4 = a0
r5 = a1
r6 = a2
r7 = a3
r8 = a4
r9 = a5
r10 = a6
r11 = a7
r12 = t0
r13 = t1
r14 = t2
r15 = t3
r16 = t4
r17 = t5
r18 = t6
r19 = t7
r20 = t8
r22 = fp
r23 = s0
r24 = s1
r25 = s2
r26 = s3
r27 = s4
r28 = s5
r29 = s6
r30 = s7
r31 = s8

f0 = fa0
f1 = fa1
f2 = fa2
f3 = fa3
f4 = fa4
f5 = fa5
f6 = fa6
f7 = fa7
f8 = ft0
f9 = ft1
f10 = ft2
f11 = ft3
f12 = ft4
f13 = ft5
f14 = ft6
f15 = ft7
f16 = ft8
f17 = ft9
f18 = ft10
f19 = ft11
f20 = ft12
f21 = ft13
f22 = ft14
f23 = ft15
f24 = fs0
f25 = fs1
f26 = fs2
f27 = fs3
f28 = fs4
f29 = fs5
f30 = fs6
f31 = fs7
s9 = fp

# Register number -> register, for gpr(n) and fpr(n).
_r = (zero, ra, tp, sp, a0, a1, a2, a3, a4, a5, a6, a7, t0, t1, t2, t3, t4, t5, t6, t7, t8, r21, fp, s0, s1, s2, s3, s4, s5, s6, s7, s8)
_f = (fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7, ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, fs0, fs1, fs2, fs3, fs4, fs5, fs6, fs7)

def _pick[T: Reg](table: tuple[T, ...], n: int, what: str) -> T:
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < len(table):
        raise EncodeError(f"{what}: register number {n!r} out of range")
    return table[n]


def gpr(n: int) -> R:
    """Return the integer register r<n> (n in 0..31, 0 is zero)."""
    return _pick(_r, n, "gpr")


def fpr(n: int) -> F:
    """Return the floating point register f<n> (n in 0..31)."""
    return _pick(_f, n, "fpr")


ALL_REGS: dict[str, Reg] = {
    r.name: r
    for r in (*_r, *_f, fcc0, fcc1, fcc2, fcc3, fcc4, fcc5, fcc6, fcc7, fcsr0, fcsr1, fcsr2, fcsr3)
}

__all__ = [
    "Reg", "R", "F", "Fcc", "Fcsr", "gpr", "fpr", "s9", "zero", "ra", "tp", "sp", "a0", "a1",
    "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
    "r21", "fp", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "fa0", "fa1", "fa2",
    "fa3", "fa4", "fa5", "fa6", "fa7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "ft8", "ft9", "ft10", "ft11", "ft12", "ft13", "ft14", "ft15", "fs0", "fs1", "fs2", "fs3",
    "fs4", "fs5", "fs6", "fs7", "fcc0", "fcc1", "fcc2", "fcc3", "fcc4", "fcc5", "fcc6", "fcc7",
    "fcsr0", "fcsr1", "fcsr2", "fcsr3", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8",
    "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r22",
    "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31", "f0", "f1", "f2", "f3", "f4",
    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18",
    "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
]
