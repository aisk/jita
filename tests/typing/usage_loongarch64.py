"""Accepted loongarch64 code beyond single calls, typed views included.
Must type check cleanly (tests/test_typing.py)."""

import ctypes

from jita import Assembler, Extern, Label
from jita.loongarch64 import *  # noqa: F403
from jita.loongarch64 import F, Fcc, Loongarch64Assembler, R
from jita.loongarch64.structs import Typed, TypedArray


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("tag", ctypes.c_uint8), ("v", ctypes.c_double * 4)]


def load_pair(dst: R, src: R, base: R) -> None:
    ld.d(dst, base, 0)
    ld.d(src, base, 8)


def compare(flag: Fcc, x: F, y: F) -> None:
    fcmp.clt.d(flag, x, y)
    bcnez(flag, "less")


def branch(target: Label) -> None:
    bnez(a0, target)
    beq(a0, a1, "done")


def untyped(dst, src):
    move(dst, src)


def with_assembler(a: Loongarch64Assembler) -> None:
    a.add.d(a0, a1, a2)
    a.fcmp.sune.s(fcc1, fa0, fa1)
    a.amadd_db.w(a0, a1, a2)
    a.crc.w.b.w(a0, a1, a2)
    a.revb._2h(a0, a1)
    a.break_(0)
    a.ret()


a = Assembler("loongarch64")
with a:
    label("entry")
    addi.d(sp, sp, -16)
    st.d(ra, sp, 8)
    li.d(a0, 0x123456789ABCDEF0)
    li.w(a1, -1)
    la.local(a1, "entry")
    ld.d(t0, a.extern_slot(Extern("puts")))
    jirl(ra, t0, 0)
    call36(Extern("puts"))
    tail36(t0, "entry")
    st.w(a0, "entry", t1)
    fld.d(fa0, "entry", t1)
    ll.w(a0, a1, 0)
    sc.w(a0, a1, 0)
    bstrpick.d(a0, a1, 31, 0)
    alsl.d(a0, a1, a2, 3)
    movgr2fr.d(fa0, a0)
    movgr2fcsr(fcsr0, a0)
    fsel(fa0, fa1, fa2, fcc0)
    and_(a0, a1, a2)
    or_(a0, a1, zero)
    assert r4 is gpr(4)
    assert fs0 is fpr(24)
    b("entry")
# Typed views: fields are Any, `addr` is an unsized MemExpr.
p: Typed = typed(a0, Point)
ld.w(a1, p.x)
ld.bu(a2, p.tag)
fld.d(fa0, p.v[1])
st.w(a1, p["x"])
amadd_db.w(a1, a2, p.x)
ldgt.w(a1, p.x, a3)
start: MemExpr = p.addr
ld.d(a1, start)
arr: TypedArray = typed(a2 + 8, ctypes.c_int32 * 4)
ldx.bu(a1, typed(a2, ctypes.c_uint8 * 4)[a3])
ld.w(a1, typed(sp - 16, ctypes.c_int32))
