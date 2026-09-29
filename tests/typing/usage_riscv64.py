"""Accepted riscv64 code beyond single calls, typed views included. Must
type check cleanly (tests/test_typing.py)."""

import ctypes

from jita import Assembler, Extern, Label
from jita.riscv64 import *  # noqa: F403
from jita.riscv64 import CsrName, F, FenceSet, Riscv64Assembler, Rm, X
from jita.riscv64.structs import Typed, TypedArray


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("tag", ctypes.c_uint8), ("v", ctypes.c_double * 4)]


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
# Typed views: fields are Any, `addr` is an unsized MemExpr.
p: Typed = typed(a0, Point)
lw(a1, p.x)
lbu(a2, p.tag)
fld(fa0, p.v[1])
sw(a1, p["x"])
amoadd.w(a1, a2, p.x)
start: MemExpr = p.addr
ld(a1, start)
arr: TypedArray = typed(a2 + 8, ctypes.c_int32 * 4)
lw(a1, arr[3])
lw(a1, typed(mem[sp + 16], ctypes.c_int32))
