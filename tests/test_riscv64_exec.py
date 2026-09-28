"""Execution tests for riscv64 code. They need a riscv64 host (or qemu-user)
and are skipped elsewhere; encodings are covered by test_riscv64_encode.py.

Set JITA_REQUIRE_RISCV64=1 to make a non-riscv64 host an error instead of
a skip, so a CI job cannot pass without executing them."""

import ctypes
import math
import os
import platform
import random
import struct

import pytest

import jita.riscv64 as R
from jita import Assembler, Extern
from jita.riscv64 import *  # noqa: F403

HOST = platform.machine().lower() == "riscv64" and platform.system() != "Windows"

if os.environ.get("JITA_REQUIRE_RISCV64") and not HOST:
    raise RuntimeError(f"JITA_REQUIRE_RISCV64 is set but the host is {platform.machine()}")

needs_riscv64_host = pytest.mark.skipif(not HOST, reason="needs a riscv64 POSIX host")

pytestmark = needs_riscv64_host


def test_add_two_ints():
    a = Assembler(R)
    with a:
        addw(a0, a0, a1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


def test_host_default_arch():
    assert Assembler().arch is R.ARCH
    assert type(Assembler()) is R.Riscv64Assembler


def test_sum_loop():
    a = Assembler()
    with a:
        mv(a2, zero)  # noqa: F405
        beqz(a1, "done")  # noqa: F405
        label("loop")  # noqa: F405
        ld(a3, mem[a0])  # noqa: F405
        add(a2, a2, a3)  # noqa: F405
        addi(a0, a0, 8)  # noqa: F405
        addi(a1, a1, -1)  # noqa: F405
        bnez(a1, "loop")  # noqa: F405
        label("done")  # noqa: F405
        mv(a0, a2)  # noqa: F405
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
        addi(sp, sp, -16)  # noqa: F405
        sd(ra, mem[sp + 8])  # noqa: F405
        ld(t0, a.extern_slot(Extern("strlen")))  # noqa: F405
        jalr(t0)  # noqa: F405
        addi(a0, a0, 1)  # noqa: F405
        ld(ra, mem[sp + 8])  # noqa: F405
        addi(sp, sp, 16)  # noqa: F405
        ret()  # noqa: F405
    with a.load(externs={"strlen": strlen}) as mod:
        assert mod.function(ctypes.c_size_t, ctypes.c_char_p)(b"hello") == 6


def test_double_math():
    a = Assembler()
    with a:
        # (x + y) * 0.5 + sqrt(y), with 0.5 built by li and fmv.d.x
        fadd.d(fa0, fa0, fa1)  # noqa: F405
        li(t0, struct.unpack("<q", struct.pack("<d", 0.5))[0])  # noqa: F405
        fmv.d.x(ft0, t0)  # noqa: F405
        fsqrt.d(fa1, fa1)  # noqa: F405
        fmadd.d(fa0, fa0, ft0, fa1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        fn = mod.function(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        assert fn(3.0, 25.0) == 14.0 + 5.0
        assert math.isnan(fn(1.0, -1.0))


def test_float_conversion_rounding_modes():
    a = Assembler()
    with a:
        label("trunc")  # noqa: F405
        fcvt.l.d(a0, fa0, rm="rtz")  # noqa: F405
        ret()  # noqa: F405
        label("up")  # noqa: F405
        fcvt.l.d(a0, fa0, rm="rup")  # noqa: F405
        ret()  # noqa: F405
        label("single")  # noqa: F405
        fcvt.s.l(ft0, a0)  # noqa: F405
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
    values = [0, -1, 2047, -2048, 0x800, 0x7FFFFFFF, -0x80000000, 0x80000000, 0xFFFFFFFF,
              0x123456789ABCDEF0, (1 << 63) - 1, -(1 << 63), (1 << 64) - 1]  # fmt: skip
    values += [rng.getrandbits(64) for _ in range(200)]
    values += [1 << rng.randrange(64) for _ in range(50)]
    a = Assembler()
    with a:
        for i, v in enumerate(values):
            li(t0, v)  # noqa: F405
            sd(t0, mem[a0 + 8 * (i % 256)])  # noqa: F405
            if i % 256 == 255:
                addi(a0, a0, 2040)  # noqa: F405
                addi(a0, a0, 8)  # noqa: F405
        ret()  # noqa: F405
    out = (ctypes.c_uint64 * len(values))()
    with a.load() as mod:
        mod.function(None, ctypes.c_void_p)(ctypes.addressof(out))
    assert list(out) == [v & ((1 << 64) - 1) for v in values]


def test_call_and_tail_local_functions():
    a = Assembler()
    with a:
        addi(sp, sp, -16)  # noqa: F405
        sd(ra, mem[sp + 8])  # noqa: F405
        call("double")  # noqa: F405
        call("double")  # noqa: F405
        ld(ra, mem[sp + 8])  # noqa: F405
        addi(sp, sp, 16)  # noqa: F405
        tail("inc")  # noqa: F405 - returns to our caller
        label("double")  # noqa: F405
        slli(a0, a0, 1)  # noqa: F405
        ret()  # noqa: F405
        label("inc")  # noqa: F405
        addi(a0, a0, 1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int64, ctypes.c_int64)(5) == 21


def test_amoadd_returns_old_value():
    a = Assembler()
    with a:
        amoadd.d.aqrl(a0, a1, mem[a0])  # noqa: F405
        ret()  # noqa: F405
    cell = ctypes.c_int64(40)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
        assert fn(ctypes.addressof(cell), 2) == 40
        assert fn(ctypes.addressof(cell), -50) == 42
    assert cell.value == -8


def test_lr_sc_increment_loop():
    a = Assembler()
    with a:
        label("retry")  # noqa: F405
        lr.w.aq(t0, mem[a0])  # noqa: F405
        addw(t0, t0, a1)  # noqa: F405
        sc.w.rl(t1, t0, mem[a0])  # noqa: F405
        bnez(t1, "retry")  # noqa: F405
        mv(a0, t0)  # noqa: F405
        ret()  # noqa: F405
    cell = ctypes.c_int32(0x7FFFFFFF)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32)
        assert fn(ctypes.addressof(cell), 1) == -0x80000000
        assert fn(ctypes.addressof(cell), 5) == -0x7FFFFFFB
    assert cell.value == -0x7FFFFFFB


def test_writable_section_and_label_loads():
    a = Assembler()
    with a:
        label("bump")  # noqa: F405
        ld(t0, "counter")  # noqa: F405
        addi(t0, t0, 1)  # noqa: F405
        sd(t0, "counter", t1)  # noqa: F405
        lla(a1, "counter")  # noqa: F405
        ld(a0, mem[a1])  # noqa: F405
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


def test_zba_zbb_and_compare_branches():
    a = Assembler()
    with a:
        # cpop(a0) + (a1 << 3 + a0) + (bgt a0, a1 ? 1000 : 0), and rev8 on the side
        cpop(t0, a0)  # noqa: F405
        sh3add(t1, a1, a0)  # noqa: F405
        add(t0, t0, t1)  # noqa: F405
        rev8(t2, a0)  # noqa: F405
        sd(t2, mem[a2])  # noqa: F405
        ble(a0, a1, "skip")  # noqa: F405
        addi(t0, t0, 1000)  # noqa: F405
        label("skip")  # noqa: F405
        mv(a0, t0)  # noqa: F405
        ret()  # noqa: F405
    out = ctypes.c_uint64()
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p)
        assert fn(0xFF, 2, ctypes.addressof(out)) == 8 + 16 + 0xFF + 1000
        assert out.value == 0xFF00_0000_0000_0000
        assert fn(1, 2, ctypes.addressof(out)) == 1 + 16 + 1


def test_rounding_mode_csr():
    a = Assembler()
    with a:
        frrm(t1)  # noqa: F405 - keep the caller's rounding mode
        fsrm(a0)  # noqa: F405
        fcvt.l.d(a0, fa0)  # noqa: F405 - dyn: uses frm
        fsrm(t1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_int64, ctypes.c_double)
        assert [fn(rm, 2.5) for rm in (0, 1, 2, 3)] == [2, 2, 2, 3]


def test_icache_flush_paths():
    # _find_flush (what load uses) takes libgcc's __clear_cache when
    # libgcc_s is installed and the system call otherwise; the fallback is
    # also called directly, so both paths run where libgcc_s exists. The
    # CI job installs it and sets JITA_REQUIRE_RISCV64, so there the libgcc
    # path must be found.
    if os.environ.get("JITA_REQUIRE_RISCV64"):
        assert ctypes.CDLL("libgcc_s.so.1").__clear_cache
    buf = ctypes.create_string_buffer(64)
    addr = ctypes.addressof(buf)
    R._find_flush()(addr, addr + 64)
    R._syscall_flush()(addr, addr + 64)
