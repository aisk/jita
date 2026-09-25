"""x64 register objects.

Every register is an instance of a width class: `Gp8`, `Gp16`, `Gp32`,
`Gp64` (general purpose), `Xmm`, `Ymm`, `St` (x87 stack) and `Rip`. The
classes are ordinary `Reg` subclasses, so helpers can annotate their
parameters with them and type checkers can tell `add(rax, ecx)` apart
from `add(rax, rcx)`.
"""

from typing import TYPE_CHECKING

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Register

if TYPE_CHECKING:
    from .mem import MemAny, MemExpr


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

    def _mem(self) -> MemAny:
        from .mem import MemAny

        return MemAny(base=self)

    def __add__(self, other: int | Reg | MemExpr | Label | Extern | str) -> MemAny:
        return self._mem() + other

    def __radd__(self, other: int | Reg | MemExpr | Label | Extern | str) -> MemAny:
        return self._mem() + other

    def __sub__(self, other: int) -> MemAny:
        return self._mem() - other

    def __mul__(self, scale: int) -> MemAny:
        from .mem import MemAny

        if not isinstance(scale, int) or isinstance(scale, bool):
            return NotImplemented
        return MemAny(index=self, scale=scale)

    def __rmul__(self, scale: int) -> MemAny:
        return self.__mul__(scale)


class _Fixed(Reg):
    # A register class with a fixed kind and size.

    __slots__ = ()
    KIND = ""
    SIZE = 0

    def __init__(self, name: str, code: int, high: bool = False):
        super().__init__(name, self.KIND, code, self.SIZE, high)


class Gp8(_Fixed):
    """8 bit general purpose register: al..r15b, spl..dil, ah..bh."""

    __slots__ = ()
    KIND, SIZE = "gp", 1


class Gp16(_Fixed):
    """16 bit general purpose register: ax..r15w."""

    __slots__ = ()
    KIND, SIZE = "gp", 2


class Gp32(_Fixed):
    """32 bit general purpose register: eax..r15d."""

    __slots__ = ()
    KIND, SIZE = "gp", 4


class Gp64(_Fixed):
    """64 bit general purpose register: rax..r15."""

    __slots__ = ()
    KIND, SIZE = "gp", 8


class Xmm(_Fixed):
    """128 bit SSE register: xmm0..xmm15."""

    __slots__ = ()
    KIND, SIZE = "xmm", 16


class Ymm(_Fixed):
    """256 bit AVX register: ymm0..ymm15."""

    __slots__ = ()
    KIND, SIZE = "ymm", 32


class St(_Fixed):
    """x87 stack register: st0..st7."""

    __slots__ = ()
    KIND, SIZE = "st", 10


class Rip(_Fixed):
    """The instruction pointer, only valid as a memory base."""

    __slots__ = ()
    KIND, SIZE = "rip", 8


# Any general purpose register, for annotating helpers.
type Gp = Gp8 | Gp16 | Gp32 | Gp64

al = Gp8("al", 0)
cl = Gp8("cl", 1)
dl = Gp8("dl", 2)
bl = Gp8("bl", 3)
spl = Gp8("spl", 4)
bpl = Gp8("bpl", 5)
sil = Gp8("sil", 6)
dil = Gp8("dil", 7)
r8b = Gp8("r8b", 8)
r9b = Gp8("r9b", 9)
r10b = Gp8("r10b", 10)
r11b = Gp8("r11b", 11)
r12b = Gp8("r12b", 12)
r13b = Gp8("r13b", 13)
r14b = Gp8("r14b", 14)
r15b = Gp8("r15b", 15)

ah = Gp8("ah", 4, high=True)
ch = Gp8("ch", 5, high=True)
dh = Gp8("dh", 6, high=True)
bh = Gp8("bh", 7, high=True)

ax = Gp16("ax", 0)
cx = Gp16("cx", 1)
dx = Gp16("dx", 2)
bx = Gp16("bx", 3)
sp = Gp16("sp", 4)
bp = Gp16("bp", 5)
si = Gp16("si", 6)
di = Gp16("di", 7)
r8w = Gp16("r8w", 8)
r9w = Gp16("r9w", 9)
r10w = Gp16("r10w", 10)
r11w = Gp16("r11w", 11)
r12w = Gp16("r12w", 12)
r13w = Gp16("r13w", 13)
r14w = Gp16("r14w", 14)
r15w = Gp16("r15w", 15)

eax = Gp32("eax", 0)
ecx = Gp32("ecx", 1)
edx = Gp32("edx", 2)
ebx = Gp32("ebx", 3)
esp = Gp32("esp", 4)
ebp = Gp32("ebp", 5)
esi = Gp32("esi", 6)
edi = Gp32("edi", 7)
r8d = Gp32("r8d", 8)
r9d = Gp32("r9d", 9)
r10d = Gp32("r10d", 10)
r11d = Gp32("r11d", 11)
r12d = Gp32("r12d", 12)
r13d = Gp32("r13d", 13)
r14d = Gp32("r14d", 14)
r15d = Gp32("r15d", 15)

rax = Gp64("rax", 0)
rcx = Gp64("rcx", 1)
rdx = Gp64("rdx", 2)
rbx = Gp64("rbx", 3)
rsp = Gp64("rsp", 4)
rbp = Gp64("rbp", 5)
rsi = Gp64("rsi", 6)
rdi = Gp64("rdi", 7)
r8 = Gp64("r8", 8)
r9 = Gp64("r9", 9)
r10 = Gp64("r10", 10)
r11 = Gp64("r11", 11)
r12 = Gp64("r12", 12)
r13 = Gp64("r13", 13)
r14 = Gp64("r14", 14)
r15 = Gp64("r15", 15)

xmm0 = Xmm("xmm0", 0)
xmm1 = Xmm("xmm1", 1)
xmm2 = Xmm("xmm2", 2)
xmm3 = Xmm("xmm3", 3)
xmm4 = Xmm("xmm4", 4)
xmm5 = Xmm("xmm5", 5)
xmm6 = Xmm("xmm6", 6)
xmm7 = Xmm("xmm7", 7)
xmm8 = Xmm("xmm8", 8)
xmm9 = Xmm("xmm9", 9)
xmm10 = Xmm("xmm10", 10)
xmm11 = Xmm("xmm11", 11)
xmm12 = Xmm("xmm12", 12)
xmm13 = Xmm("xmm13", 13)
xmm14 = Xmm("xmm14", 14)
xmm15 = Xmm("xmm15", 15)

ymm0 = Ymm("ymm0", 0)
ymm1 = Ymm("ymm1", 1)
ymm2 = Ymm("ymm2", 2)
ymm3 = Ymm("ymm3", 3)
ymm4 = Ymm("ymm4", 4)
ymm5 = Ymm("ymm5", 5)
ymm6 = Ymm("ymm6", 6)
ymm7 = Ymm("ymm7", 7)
ymm8 = Ymm("ymm8", 8)
ymm9 = Ymm("ymm9", 9)
ymm10 = Ymm("ymm10", 10)
ymm11 = Ymm("ymm11", 11)
ymm12 = Ymm("ymm12", 12)
ymm13 = Ymm("ymm13", 13)
ymm14 = Ymm("ymm14", 14)
ymm15 = Ymm("ymm15", 15)

st0 = St("st0", 0)
st1 = St("st1", 1)
st2 = St("st2", 2)
st3 = St("st3", 3)
st4 = St("st4", 4)
st5 = St("st5", 5)
st6 = St("st6", 6)
st7 = St("st7", 7)

# rip is only valid as a memory base. Its code is the ModRM r/m value (101)
# that selects rip-relative addressing with mod=00.
rip = Rip("rip", 5)

# Register number -> register, for gp8(n) and friends.
_gp8 = (al, cl, dl, bl, spl, bpl, sil, dil, r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b)
_gp16 = (ax, cx, dx, bx, sp, bp, si, di, r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w)
_gp32 = (eax, ecx, edx, ebx, esp, ebp, esi, edi, r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d)
_gp64 = (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15)
_xmm = (xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15)
_ymm = (ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15)
_st = (st0, st1, st2, st3, st4, st5, st6, st7)


def _pick[R: Reg](table: tuple[R, ...], n: int, what: str) -> R:
    if not isinstance(n, int) or not 0 <= n < len(table):
        raise EncodeError(f"{what}: register number {n!r} out of range")
    return table[n]


def gp8(n: int) -> Gp8:
    """Return the gp8 register with hardware number n (0..15)."""
    return _pick(_gp8, n, "gp8")


def gp16(n: int) -> Gp16:
    """Return the gp16 register with hardware number n (0..15)."""
    return _pick(_gp16, n, "gp16")


def gp32(n: int) -> Gp32:
    """Return the gp32 register with hardware number n (0..15)."""
    return _pick(_gp32, n, "gp32")


def gp64(n: int) -> Gp64:
    """Return the gp64 register with hardware number n (0..15)."""
    return _pick(_gp64, n, "gp64")


def xmm(n: int) -> Xmm:
    """Return the xmm register with hardware number n (0..15)."""
    return _pick(_xmm, n, "xmm")


def ymm(n: int) -> Ymm:
    """Return the ymm register with hardware number n (0..15)."""
    return _pick(_ymm, n, "ymm")


def st(n: int) -> St:
    """Return the x87 stack register st(n) (n in 0..7)."""
    return _pick(_st, n, "st")


ALL_REGS: dict[str, Reg] = {
    r.name: r for r in (*_gp8, ah, ch, dh, bh, *_gp16, *_gp32, *_gp64, *_xmm, *_ymm, *_st, rip)
}

__all__ = [
    "Reg", "Gp8", "Gp16", "Gp32", "Gp64", "Xmm", "Ymm", "St", "Rip", "Gp", "gp8",
    "gp16", "gp32", "gp64", "xmm", "ymm", "st", "al", "cl", "dl", "bl", "spl", "bpl",
    "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b", "ah",
    "ch", "dh", "bh", "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w",
    "r10w", "r11w", "r12w", "r13w", "r14w", "r15w", "eax", "ecx", "edx", "ebx", "esp",
    "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
    "r12", "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
    "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9",
    "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "st0", "st1", "st2", "st3",
    "st4", "st5", "st6", "st7", "rip",
]
