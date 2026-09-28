"""Execution tests for loongarch64 code. They need a loongarch64 host (or
qemu-user) and are skipped elsewhere; encodings are covered by
test_loongarch64_encode.py.

Set JITA_REQUIRE_LOONGARCH64=1 to make a non-loongarch64 host an error
instead of a skip, so a CI job cannot pass without executing them."""

import binascii
import ctypes
import math
import os
import platform
import random
import struct

import pytest

import jita.loongarch64 as L
from jita import Assembler, Extern
from jita.loongarch64 import *  # noqa: F403

HOST = platform.machine().lower() == "loongarch64" and platform.system() != "Windows"

if os.environ.get("JITA_REQUIRE_LOONGARCH64") and not HOST:
    raise RuntimeError(f"JITA_REQUIRE_LOONGARCH64 is set but the host is {platform.machine()}")

needs_loongarch64_host = pytest.mark.skipif(not HOST, reason="needs a loongarch64 POSIX host")

pytestmark = needs_loongarch64_host


def test_add_two_ints():
    a = Assembler(L)
    with a:
        add.w(a0, a0, a1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


def test_host_default_arch():
    assert Assembler().arch is L.ARCH
    assert type(Assembler()) is L.Loongarch64Assembler


def test_sum_loop():
    a = Assembler()
    with a:
        move(a2, zero)  # noqa: F405
        beqz(a1, "done")  # noqa: F405
        label("loop")  # noqa: F405
        ld.d(a3, a0, 0)  # noqa: F405
        add.d(a2, a2, a3)  # noqa: F405
        addi.d(a0, a0, 8)  # noqa: F405
        addi.d(a1, a1, -1)  # noqa: F405
        bnez(a1, "loop")  # noqa: F405
        label("done")  # noqa: F405
        move(a0, a2)  # noqa: F405
        ret()  # noqa: F405
    vals = (ctypes.c_int64 * 100)(*range(1, 101))
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
        assert fn(vals, 100) == 5050
        assert fn(vals, 0) == 0


def test_call_extern_through_slot():
    libc = ctypes.CDLL(None)
    strlen = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    assert strlen is not None
    a = Assembler()
    with a:
        addi.d(sp, sp, -16)  # noqa: F405
        st.d(ra, sp, 8)  # noqa: F405
        ld.d(t0, a.extern_slot(Extern("strlen")))  # noqa: F405
        jirl(ra, t0, 0)  # noqa: F405
        addi.d(a0, a0, 1)  # noqa: F405
        ld.d(ra, sp, 8)  # noqa: F405
        addi.d(sp, sp, 16)  # noqa: F405
        ret()  # noqa: F405
    with a.load(externs={"strlen": strlen}) as mod:
        assert mod.function(ctypes.c_size_t, ctypes.c_char_p)(b"hello") == 6


def test_double_math():
    a = Assembler()
    with a:
        # (x + y) * 0.5 + sqrt(y), with 0.5 built by li.d and movgr2fr.d
        fadd.d(fa0, fa0, fa1)  # noqa: F405
        li.d(t0, struct.unpack("<q", struct.pack("<d", 0.5))[0])  # noqa: F405
        movgr2fr.d(ft0, t0)  # noqa: F405
        fsqrt.d(fa1, fa1)  # noqa: F405
        fmadd.d(fa0, fa0, ft0, fa1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        fn = mod.function(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        assert fn(3.0, 25.0) == 14.0 + 5.0
        assert math.isnan(fn(1.0, -1.0))


def test_float_conversions():
    a = Assembler()
    with a:
        label("trunc")  # noqa: F405
        ftintrz.l.d(ft0, fa0)  # noqa: F405
        movfr2gr.d(a0, ft0)  # noqa: F405
        ret()  # noqa: F405
        label("up")  # noqa: F405
        ftintrp.l.d(ft0, fa0)  # noqa: F405
        movfr2gr.d(a0, ft0)  # noqa: F405
        ret()  # noqa: F405
        label("single")  # noqa: F405
        movgr2fr.d(ft0, a0)  # noqa: F405
        ffint.s.l(ft0, ft0)  # noqa: F405
        fcvt.d.s(fa0, ft0)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        trunc = mod.function(ctypes.c_int64, ctypes.c_double, entry="trunc")
        up = mod.function(ctypes.c_int64, ctypes.c_double, entry="up")
        single = mod.function(ctypes.c_double, ctypes.c_int64, entry="single")
        assert (trunc(2.7), trunc(-2.7), up(2.1), up(-2.9)) == (2, -2, 3, -2)
        assert single(1 << 40) == float(1 << 40)


def test_li_big_constants():
    rng = random.Random(7)
    values = [0, -1, 0x7FF, 0x800, -0x800, 0xFFF, 0x7FFFFFFF, -0x80000000, 0x80000000, 0xFFFFFFFF,
              0x0010000000000000, 0x123456789ABCDEF0, (1 << 63) - 1, -(1 << 63), (1 << 64) - 1]  # fmt: skip
    values += [rng.getrandbits(64) for _ in range(200)]
    values += [1 << rng.randrange(64) for _ in range(50)]
    words = [0x7FF, 0x800, -1, 0xFFFFFFFF, -0x80000000, 0x12345678] + [rng.getrandbits(32) for _ in range(40)]
    a = Assembler()
    with a:
        for i, v in enumerate(values):
            li.d(t0, v)  # noqa: F405
            stptr.d(t0, a0, 8 * i)  # noqa: F405
        for i, v in enumerate(words):
            li.w(t0, v)  # noqa: F405
            stptr.d(t0, a1, 8 * i)  # noqa: F405
        ret()  # noqa: F405
    out = (ctypes.c_uint64 * len(values))()
    out_w = (ctypes.c_uint64 * len(words))()
    with a.load() as mod:
        mod.function(None, ctypes.c_void_p, ctypes.c_void_p)(ctypes.addressof(out), ctypes.addressof(out_w))
    assert list(out) == [v & ((1 << 64) - 1) for v in values]
    sext = [(v & 0xFFFFFFFF) - ((v & 0x80000000) << 1) for v in words]
    assert list(out_w) == [v & ((1 << 64) - 1) for v in sext]


def test_call36_and_tail36_local_functions():
    a = Assembler()
    with a:
        addi.d(sp, sp, -16)  # noqa: F405
        st.d(ra, sp, 8)  # noqa: F405
        call36("double")  # noqa: F405
        call("double")  # noqa: F405
        ld.d(ra, sp, 8)  # noqa: F405
        addi.d(sp, sp, 16)  # noqa: F405
        tail36(t0, "inc")  # noqa: F405 - returns to our caller
        label("double")  # noqa: F405
        slli.d(a0, a0, 1)  # noqa: F405
        ret()  # noqa: F405
        label("inc")  # noqa: F405
        addi.d(a0, a0, 1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int64, ctypes.c_int64)(5) == 21


def test_amadd_returns_old_value():
    a = Assembler()
    with a:
        amadd_db.d(t0, a1, a0)  # noqa: F405
        move(a0, t0)  # noqa: F405
        ret()  # noqa: F405
    cell = ctypes.c_int64(40)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
        assert fn(ctypes.addressof(cell), 2) == 40
        assert fn(ctypes.addressof(cell), -50) == 42
    assert cell.value == -8


def test_ll_sc_increment_loop():
    a = Assembler()
    with a:
        label("retry")  # noqa: F405
        ll.w(t0, a0, 0)  # noqa: F405
        add.w(t1, t0, a1)  # noqa: F405
        move(t2, t1)  # noqa: F405
        sc.w(t1, a0, 0)  # noqa: F405 - t1 becomes 1 on success
        beqz(t1, "retry")  # noqa: F405
        dbar(0)  # noqa: F405
        move(a0, t2)  # noqa: F405
        ret()  # noqa: F405
    cell = ctypes.c_int32(0x7FFFFFFF)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32)
        assert fn(ctypes.addressof(cell), 1) == -0x80000000
        assert fn(ctypes.addressof(cell), 5) == -0x7FFFFFFB
    assert cell.value == -0x7FFFFFFB


def test_fcmp_and_condition_branches():
    a = Assembler()
    with a:
        # max(x, y) through fcmp.clt.d + bceqz, and 1.0 / 0.0 selected by fsel
        label("max")  # noqa: F405
        fcmp.clt.d(fcc0, fa0, fa1)  # noqa: F405
        bceqz(fcc0, "keep")  # noqa: F405
        fmov.d(fa0, fa1)  # noqa: F405
        label("keep")  # noqa: F405
        ret()  # noqa: F405
        label("unordered")  # noqa: F405
        fcmp.cun.d(fcc3, fa0, fa1)  # noqa: F405
        movcf2gr(a0, fcc3)  # noqa: F405
        ret()  # noqa: F405
        label("sgt")  # noqa: F405
        fcmp.sgt.s(fcc7, fa0, fa1)  # noqa: F405 - fcmp.slt.s with the operands swapped
        bcnez(fcc7, "yes")  # noqa: F405
        move(a0, zero)  # noqa: F405
        ret()  # noqa: F405
        label("yes")  # noqa: F405
        li.w(a0, 1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        mx = mod.function(ctypes.c_double, ctypes.c_double, ctypes.c_double, entry="max")
        unordered = mod.function(ctypes.c_int64, ctypes.c_double, ctypes.c_double, entry="unordered")
        sgt = mod.function(ctypes.c_int64, ctypes.c_float, ctypes.c_float, entry="sgt")
        assert (mx(1.0, 2.0), mx(3.0, -2.0)) == (2.0, 3.0)
        assert (unordered(1.0, math.nan), unordered(1.0, 2.0)) == (1, 0)
        assert (sgt(2.0, 1.0), sgt(1.0, 2.0), sgt(1.0, 1.0)) == (1, 0, 0)


def test_writable_section_and_label_loads():
    a = Assembler()
    with a:
        label("bump")  # noqa: F405
        ld.d(t0, "counter")  # noqa: F405
        addi.d(t0, t0, 1)  # noqa: F405
        st.d(t0, "counter", t1)  # noqa: F405
        la.local(a1, "counter")  # noqa: F405
        ld.d(a0, a1, 0)  # noqa: F405
        pcaddi(a2, "counter")  # noqa: F405
        ld.d(a2, a2, 0)  # noqa: F405
        sub.d(a2, a2, a0)  # noqa: F405
        add.d(a0, a0, a2)  # noqa: F405 - adds 0 when pcaddi agrees
        ret()  # noqa: F405
        with a.section("vars", writable=True):
            a.align(8)
            label("counter")  # noqa: F405
            a.qword(41)
    with a.load() as mod:
        bump = mod.function(ctypes.c_int64, entry="bump")
        assert bump() == 42
        assert bump() == 43
        assert struct.unpack("<q", ctypes.string_at(mod.address("counter"), 8))[0] == 43


def test_bit_operations_and_crc():
    a = Assembler()
    with a:
        # bstrpick.d, revb.d, alsl.d, clz.d, maskeqz and crc.w.b.w into a buffer
        label("bits")  # noqa: F405
        bstrpick.d(t0, a0, 15, 8)  # noqa: F405
        st.d(t0, a1, 0)  # noqa: F405
        revb.d(t0, a0)  # noqa: F405
        st.d(t0, a1, 8)  # noqa: F405
        alsl.d(t0, a0, a0, 3)  # noqa: F405 - a0 * 9
        st.d(t0, a1, 16)  # noqa: F405
        clz.d(t0, a0)  # noqa: F405
        st.d(t0, a1, 24)  # noqa: F405
        maskeqz(t0, a0, zero)  # noqa: F405
        st.d(t0, a1, 32)  # noqa: F405
        ret()  # noqa: F405
        label("crc")  # noqa: F405 - crc(a0 = crc, a1 = byte)
        crc.w.b.w(a0, a1, a0)  # noqa: F405
        ret()  # noqa: F405
    out = (ctypes.c_uint64 * 5)()
    with a.load() as mod:
        mod.function(None, ctypes.c_uint64, ctypes.c_void_p, entry="bits")(0x0123456789ABCDEF, ctypes.addressof(out))
        step = mod.function(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, entry="crc")
        state = 0xFFFFFFFF
        for byte in b"jita":
            state = step(state, byte)
    assert list(out) == [0xCD, 0xEFCDAB8967452301, (0x0123456789ABCDEF * 9) & ((1 << 64) - 1), 7, 0]
    assert state ^ 0xFFFFFFFF == binascii.crc32(b"jita")


def test_icache_flush_paths():
    # _find_flush (what load uses) takes libgcc's __clear_cache when
    # libgcc_s is installed and a JIT compiled `ibar 0` otherwise; the
    # stub is also called directly, so both paths run where libgcc_s
    # exists. The CI job installs it and sets JITA_REQUIRE_LOONGARCH64, so
    # there the libgcc path must be found.
    if os.environ.get("JITA_REQUIRE_LOONGARCH64"):
        assert ctypes.CDLL("libgcc_s.so.1").__clear_cache
    buf = ctypes.create_string_buffer(64)
    addr = ctypes.addressof(buf)
    L._find_flush()(addr, addr + 64)
    L._stub_flush()(addr, addr + 64)
