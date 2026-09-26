"""`Assembler.function` and the `function` decorator: from generated code
to a ctypes callable without handling the Module."""

import ctypes
import gc
import platform
import weakref
from typing import Any, cast

import pytest

import jita.aarch64 as aarch64
import jita.x64 as x64
from jita import Assembler, Extern, current, function
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 POSIX host",
)

I64P = ctypes.POINTER(ctypes.c_int64)
LABS = 0 if platform.system() == "Windows" else ctypes.cast(ctypes.CDLL(None).labs, ctypes.c_void_p).value or 0


def gen_sum():
    # int64_t sum(const int64_t *p, size_t n)
    xor(eax, eax)
    test(rsi, rsi)
    jz("done")
    label("loop")
    add(rax, qword[rdi])
    add(rdi, 8)
    dec(rsi)
    jnz("loop")
    label("done")
    ret()


def test_function_is_not_a_mnemonic():
    assert "function" not in x64.ARCH.insns
    assert "function" not in aarch64.ARCH.insns
    assert callable(Assembler(x64).function)


def test_assembler_function():
    a = Assembler(x64)
    with a:
        gen_sum()
    fn = a.function(ctypes.c_int64, I64P, ctypes.c_size_t)
    arr = (ctypes.c_int64 * 4)(1, 2, 3, 4)
    assert fn(arr, 4) == 10
    assert fn(arr, 0) == 0
    assert fn.assembler is a
    assert not fn.module.closed
    assert ctypes.cast(cast(Any, fn), ctypes.c_void_p).value == fn.module.image.base
    assert "add" in listing(fn.assembler, fn.module.image)


def test_assembler_function_entry():
    a = Assembler(x64)
    with a:
        mov(eax, 1)
        ret()
        a.align(16)
        label("second")
        mov(eax, 5)
        ret()
    assert a.function(ctypes.c_int)() == 1
    second = a.function(ctypes.c_int, entry="second")
    assert second() == 5
    assert ctypes.cast(cast(Any, second), ctypes.c_void_p).value == second.module.address("second")


def test_assembler_function_externs():
    a = Assembler(x64)
    with a:
        sub(rsp, 8)
        call(qword[rip + Extern("labs")])
        add(rsp, 8)
        ret()
    fn = a.function(ctypes.c_long, ctypes.c_long, externs={"labs": LABS})
    assert fn(-42) == 42


def test_independent_modules():
    a = Assembler(x64)
    with a:
        mov(eax, 3)
        ret()
    f1 = a.function(ctypes.c_int)
    f2 = a.function(ctypes.c_int)
    assert f1.module is not f2.module
    assert f1.module.image.base != f2.module.image.base
    f1.module.close()
    assert not f2.module.closed
    assert f2() == 3


def test_decorator_module_level_mnemonics():
    @function(ctypes.c_int64, I64P, ctypes.c_size_t)
    def sum_array(a):
        """Sum n int64 values."""
        assert isinstance(a, Assembler)
        gen_sum()

    arr = (ctypes.c_int64 * 3)(5, 6, 7)
    assert sum_array(arr, 3) == 18
    assert sum_array.__name__ == "sum_array"
    assert sum_array.__qualname__.endswith("<locals>.sum_array")
    assert sum_array.__doc__ == "Sum n int64 values."
    assert isinstance(sum_array.assembler, Assembler)
    assert not sum_array.module.closed


def test_decorator_assembler_methods():
    @function(ctypes.c_long, ctypes.c_long, arch="x64")
    def f(a):
        # labs(x) + table[1], with the table in a data section and the
        # extern called through its pointer slot
        with a.section("data", align=8):
            a.label("table")
            a.qword(100, 200)
        a.sub(rsp, 8)
        a.mov(rax, qword[rip + a.extern_slot(Extern("labs", LABS))])
        a.call(rax)
        a.lea(rcx, ptr[rip + "table"])
        a.add(rax, qword[rcx + 8])
        a.jmp(a.pc[0])
        a.mov(eax, 0)  # skipped
        a.label(a.pc[0])
        a.add(rsp, 8)
        a.ret()

    assert f(-5) == 205
    assert f(7) == 207
    assert "data" in f.assembler.sections
    assert "externs" in f.assembler.sections


def test_decorator_entry_and_externs():
    @function(ctypes.c_long, ctypes.c_long, entry="main", externs={"labs": LABS})
    def f():
        ud2()
        label("main")
        sub(rsp, 8)
        call(qword[rip + Extern("labs")])
        add(rsp, 8)
        ret()

    assert f(-9) == 9


def make_scaler(k: int, bias: int):
    @function(ctypes.c_int64, ctypes.c_int64)
    def scale():
        imul(rax, rdi, k)
        if bias:
            add(rax, bias)
        ret()

    return scale


def test_decorator_in_factory():
    triple = make_scaler(3, 0)
    ten_plus_one = make_scaler(10, 1)
    assert triple(7) == 21
    assert ten_plus_one(7) == 71
    assert triple.module is not ten_plus_one.module
    assert len(triple.module.image.data) != len(ten_plus_one.module.image.data)


def test_function_outlives_module_and_assembler():
    a = Assembler(x64)
    with a:
        mov(eax, 42)
        ret()
    mod = a.load()
    fn = mod.function(ctypes.c_int)
    mod_ref, asm_ref = weakref.ref(mod), weakref.ref(a)
    del mod, a
    gc.collect()
    assert asm_ref() is None
    assert mod_ref() is fn.module
    assert fn() == 42


def test_gc_releases_memory():
    a = Assembler(x64)
    with a:
        mov(eax, 42)
        ret()
    fn = a.function(ctypes.c_int)
    del a
    assert fn() == 42
    mod_ref = weakref.ref(fn.module)
    mem = fn.module.memory
    mem_ref = weakref.ref(mem)
    map_ref = weakref.ref(mem._map)
    assert not mem.closed
    del mem, fn
    gc.collect()
    assert mod_ref() is None
    assert mem_ref() is None
    assert map_ref() is None


def test_decorated_function_gc_releases_memory():
    @function(ctypes.c_int)
    def f():
        mov(eax, 1)
        ret()

    assert f() == 1
    mod_ref = weakref.ref(f.module)
    mem_ref = weakref.ref(f.module.memory)
    del f
    gc.collect()
    assert mod_ref() is None
    assert mem_ref() is None


def test_decorator_body_with_and_without_parameter():
    seen = []

    @function(ctypes.c_int)
    def with_a(a):
        seen.append(a)
        mov(eax, 1)
        ret()

    @function(ctypes.c_int)
    def without_a():
        seen.append(current())
        mov(eax, 2)
        ret()

    assert with_a() == 1 and without_a() == 2
    assert seen == [with_a.assembler, without_a.assembler]
