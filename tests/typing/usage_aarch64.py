"""Accepted aarch64 code beyond single calls, typed views included. Must
type check cleanly (tests/test_typing.py)."""

import ctypes

from jita import Assembler, Extern, Label
from jita.aarch64 import *  # noqa: F403
from jita.aarch64 import Aarch64Assembler, Cond, Gp, RegMod, X
from jita.aarch64.structs import Typed, TypedArray


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("tag", ctypes.c_uint8), ("v", ctypes.c_double * 4)]


def index(dst: X, base: X, i: RegMod[X]) -> None:
    add(dst, base, i)
    ldr(dst, mem[base + i])


def branch(cond: Cond, target: Label) -> None:
    b.ne(target)
    csel(x0, x1, x2, cond)


def clear(r: Gp) -> None:
    cbz(r, "done")


def untyped(dst, src):
    mov(dst, src)


def with_assembler(a: Aarch64Assembler) -> None:
    a.b.eq("done")
    a.movk(x0, 0x1234, lsl=16)
    a.str(x0, mem[sp + 8])
    a.ret()


a = Assembler("aarch64")
with a:
    label("entry")
    stp(x29, x30, mem.pre[sp - 16])
    ldp(x29, x30, mem.post[sp, 16])
    ldr(x0, mem[x1 + (x2 << 3)])
    ldr(w0, mem[x1 + w2.uxtw(2)])
    add(x0, x1, x2.lsl(3))
    add(x0, sp, 16)
    movz(x0, 0xFFFF, lsl=48)
    fmov(d0, 1.5)
    fadd(d0, d1, d2)
    tbz(w0, 3, "entry")
    bl(Extern("puts"))
    b.eq("entry")
    ret()
# Typed views: fields are Any, `addr` is an unsized MemExpr.
p: Typed = typed(x0, Point)
ldr(w1, p.x)
ldrb(w2, p.tag)
ldr(d0, p.v[1])
ldp(d0, d1, p.v[2])
str_(w1, p["x"])
start: MemExpr = p.addr
arr: TypedArray = typed(x2 + 8, ctypes.c_int32 * 4)
ldr(w0, typed(x2, ctypes.c_int32 * 4)[x3])
ldr(w0, typed(x2, ctypes.c_int32 * 4)[w3.uxtw()])
ldr(w0, typed(mem[sp + 16], ctypes.c_int32))
index(x0, x1, x2 << 3)
branch("ge", a.label())
clear(w3)
with_assembler(a)
a.mov(x0, gp64(3))
