"""Execution tests for aarch64 code. They need an aarch64 host and are
skipped elsewhere; encodings are covered by test_aarch64_encode.py."""

import ctypes
import platform
import struct

import pytest

import jita.aarch64 as A
from jita import Assembler, Extern
from jita.aarch64 import *  # noqa: F403

needs_aarch64_host = pytest.mark.skipif(
    platform.machine().lower() not in ("aarch64", "arm64") or platform.system() == "Windows",
    reason="needs an aarch64 POSIX host",
)

pytestmark = needs_aarch64_host


def test_add_two_ints():
    a = Assembler(A)
    with a:
        add(w0, w0, w1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


def test_host_default_arch():
    assert Assembler().arch is A.ARCH


def test_sum_loop():
    a = Assembler()
    with a:
        mov(x2, xzr)  # noqa: F405
        cbz(x1, "done")  # noqa: F405
        label("loop")  # noqa: F405
        ldr(x3, mem.post[x0, 8])  # noqa: F405
        add(x2, x2, x3)  # noqa: F405
        subs(x1, x1, 1)  # noqa: F405
        b.ne("loop")  # noqa: F405
        label("done")  # noqa: F405
        mov(x0, x2)  # noqa: F405
        ret()  # noqa: F405
    vals = (ctypes.c_int64 * 100)(*range(1, 101))
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
        assert fn(vals, 100) == 5050
        assert fn(vals, 0) == 0


def test_call_extern_through_literal():
    libc = ctypes.CDLL(None)
    strlen = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    assert strlen is not None
    a = Assembler()
    with a:
        stp(fp, lr, mem.pre[sp - 16])  # noqa: F405
        mov(fp, sp)  # noqa: F405
        ldr(x16, "ptr")  # noqa: F405
        blr(x16)  # noqa: F405
        add(x0, x0, 1)  # noqa: F405
        ldp(fp, lr, mem.post[sp, 16])  # noqa: F405
        ret()  # noqa: F405
        a.align(8)
        label("ptr")  # noqa: F405
        a.qword(Extern("strlen"))
    with a.load(externs={"strlen": strlen}) as mod:
        assert mod.function(ctypes.c_size_t, ctypes.c_char_p)(b"hello") == 6


def test_double_math_and_fmov_immediate():
    a = Assembler()
    with a:
        fadd(d0, d0, d1)  # noqa: F405
        fmov(d2, 0.5)  # noqa: F405
        fmul(d0, d0, d2)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_double, ctypes.c_double, ctypes.c_double)(3.0, 5.0) == 4.0


def test_csel_max_and_tbz():
    a = Assembler()
    with a:
        cmp(x0, x1)  # noqa: F405
        csel(x0, x0, x1, "gt")  # noqa: F405
        tbz(x0, 0, "even")  # noqa: F405
        add(x0, x0, 1000)  # noqa: F405
        label("even")  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
        assert fn(4, -9) == 4
        assert fn(-3, 7) == 1007


def test_adr_to_data_section_and_bitfields():
    a = Assembler()
    with a:
        adr(x1, "val")  # noqa: F405
        ldr(x0, mem[x1])  # noqa: F405
        ubfx(x0, x0, 8, 16)  # noqa: F405
        orr(x0, x0, 0xF0000)  # noqa: F405
        ret()  # noqa: F405
        with a.section("data"):
            label("val")  # noqa: F405
            a.qword(0x11_2233_4455)
    with a.load() as mod:
        assert mod.function(ctypes.c_uint64)() == 0xF3344


def test_writable_section_store_and_movk():
    a = Assembler()
    with a:
        movz(x1, 0xBEEF)  # noqa: F405
        movk(x1, 0xDEAD, lsl=16)  # noqa: F405
        adr(x2, "slot")  # noqa: F405
        str_(x1, mem[x2])  # noqa: F405
        ret()  # noqa: F405
    with a.section("vars", writable=True):
        a.label("slot")
        a.qword(0)
    with a.load() as mod:
        mod.function(None)()
        addr = mod.address("slot")
        assert struct.unpack("<Q", ctypes.string_at(addr, 8))[0] == 0xDEADBEEF


def test_bl_local_function():
    a = Assembler()
    with a:
        stp(fp, lr, mem.pre[sp - 16])  # noqa: F405
        bl("double")  # noqa: F405
        bl("double")  # noqa: F405
        ldp(fp, lr, mem.post[sp, 16])  # noqa: F405
        ret()  # noqa: F405
        label("double")  # noqa: F405
        lsl(x0, x0, 1)  # noqa: F405
        ret()  # noqa: F405
    with a.load() as mod:
        assert mod.function(ctypes.c_int64, ctypes.c_int64)(5) == 20
