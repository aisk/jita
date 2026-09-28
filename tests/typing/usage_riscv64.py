"""Accepted riscv64 code beyond single calls. Must type check cleanly
(tests/test_typing.py)."""

from jita import Assembler, Extern, Label
from jita.riscv64 import *  # noqa: F403
from jita.riscv64 import CsrName, F, FenceSet, Riscv64Assembler, Rm, X


def load_pair(dst: X, src: X, base: X) -> None:
    ld(dst, mem[base])
    ld(src, mem[base + 8])


def convert(dst: X, src: F, mode: Rm) -> None:
    fcvt.l.d(dst, src, rm=mode)


def barrier(pred: FenceSet, succ: FenceSet, csr: CsrName) -> None:
    fence(pred, succ)
    csrr(a0, csr)


def branch(target: Label) -> None:
    bnez(a0, target)
    beq(a0, a1, "done")


def untyped(dst, src):
    mv(dst, src)


def with_assembler(a: Riscv64Assembler) -> None:
    a.fadd.d(fa0, fa1, fa2)
    a.fcvt.w.d(a0, fa0, rm="rtz")
    a.lr.w.aq(a0, mem[a1])
    a.fence.i()
    a.min(a0, a1, a2)
    a.ret()


a = Assembler("riscv64")
with a:
    label("entry")
    addi(sp, sp, -16)
    sd(ra, mem[sp + 8])
    li(a0, 0x123456789ABCDEF0)
    lla(a1, "entry")
    ld(t0, a.extern_slot(Extern("puts")))
    jalr(t0)
    call(Extern("puts"))
    sd(a0, "entry", t0)
    fld(fa0, "entry", t0)
    amoadd.d.aqrl(a0, a1, mem[a2])
    sc.w(a0, a1, mem[a2])
    fmv.d.x(fa0, a0)
    fmv.d(fa0, fa1)
    fence()
    fence("rw", "w")
    csrrw(a0, 0x7C0, a1)
    and_(a0, a1, a2)
    min_(a0, a1, a2)
    assert x10 is gpr(10)
    assert fs0 is fpr(8)
    j("entry")
    tail("entry")
