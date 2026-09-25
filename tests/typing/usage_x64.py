"""Accepted x64 code beyond single calls: helpers, assemblers, typed
views, functions. Must type check cleanly (tests/test_typing.py)."""

import ctypes

from jita import Assembler, Extern, Label, function
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403
from jita.x64 import Gp, Gp32, Gp64, Mem64, MemAny, X64Assembler, Xmm
from jita.x64.structs import Typed


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("tag", ctypes.c_uint8), ("v", ctypes.c_double * 4)]


# Helpers annotated with the width classes, or left unannotated.
def load(dst: Gp64, src: Mem64 | MemAny) -> None:
    mov(dst, src)


def bump(r: Gp) -> None:
    inc(r)


def scale(dst: Xmm, factor: Xmm) -> None:
    mulsd(dst, factor)


def untyped(dst, src):
    mov(dst, src)
    add(dst, 1)


def with_assembler(a: X64Assembler, lbl: Label) -> None:
    a.cmp(rdi, 0)
    a.jz.short(lbl)
    a.ret()


# Operand arithmetic.
m: MemAny = rbx + rcx * 8 + 16
m2 = 8 * rcx + rbx - 4
q: Mem64 = qword[m]
r: Gp32 = gp32(3)
slot = qword[rip + Extern("strlen")]
load(rax, qword[rdi + 8])
load(rax, rdi + rsi * 8)
bump(ecx)
scale(xmm0, xmm(1))

a = Assembler("x64")
with a:
    label("entry")
    mov(rax, q)
    mov(rax, "table")
    mov(eax, r)
    call(slot)
    jmp.short("entry")
    lea(rax, ptr[rip + "table"])
    ret()
with a.section("data", align=8):
    label("table")
    a.qword(1, "entry", Extern("strlen"))
a.mov(rax, 1)
a.int(3)
a.int_(3)
a.jz.short("entry")
top: Label = a.label()
a.jmp(top)
a.align(16)
with_assembler(a, a.pc[1])

# Typed views: fields are Any, `addr` is unsized memory.
p: Typed = typed(rdi, Point)
mov(eax, p.x)
movzx(eax, p.tag)
movsd(xmm0, p.v[1])
movsd(xmm1, p.v[rcx])
lea(rax, p.addr)
arr = typed(rdi + 8, ctypes.c_int32 * 4)
mov(eax, arr[rcx])
mov(eax, typed(rsi, ctypes.c_int32))

# A plain Assembler (module argument or host) is loosely typed.
host = Assembler()
host.mov(rax, 1)


@function(ctypes.c_int64, ctypes.c_int64)
def double() -> None:
    mov(rax, rdi)
    add(rax, rax)
    ret()


result = double(21)
text: str = listing(double.assembler, double.module.image)
