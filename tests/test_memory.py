"""`ExecMemory` on every host, plus calling convention free smoke tests of
loaded code on x86-64, aarch64, riscv64 and loongarch64 hosts, Windows
included. The rest of the runtime tests write SysV code and live in
test_runtime.py."""

import ctypes
import mmap
import platform

import pytest

import jita.aarch64 as aarch64
import jita.loongarch64 as loongarch64
import jita.riscv64 as riscv64
import jita.x64 as x64
from jita import METH_O, Assembler, Extern, LoadError
from jita.runtime import ExecMemory

MACHINE = platform.machine().lower()
x86_64_host = pytest.mark.skipif(MACHINE not in ("x86_64", "amd64"), reason="needs an x86-64 host")
aarch64_host = pytest.mark.skipif(MACHINE not in ("aarch64", "arm64"), reason="needs an aarch64 host")
riscv64_host = pytest.mark.skipif(MACHINE != "riscv64", reason="needs a riscv64 host")
loongarch64_host = pytest.mark.skipif(MACHINE != "loongarch64", reason="needs a loongarch64 host")


def test_exec_memory_basics():
    with ExecMemory(10) as mem:
        assert mem.size % mmap.PAGESIZE == 0 and mem.size >= mmap.PAGESIZE
        assert mem.address % mmap.PAGESIZE == 0
        mem.write(b"\xc3")
        assert ctypes.string_at(mem.address, 1) == b"\xc3"
        with pytest.raises(LoadError):
            mem.write(b"\x00", mem.size)
        mem.protect_exec()
        with pytest.raises(LoadError):
            mem.write(b"\x90")
    assert mem.closed


def test_icache_flush_hook():
    seen = []
    mem = ExecMemory(1, icache_flush=lambda addr, size: seen.append((addr, size)))
    mem.protect_exec()
    assert seen == [(mem.address, mem.size)]
    mem.close()
    mem.close()


def test_closed_memory_rejects_use():
    mem = ExecMemory(1)
    mem.close()
    assert mem.closed
    with pytest.raises(LoadError, match="closed"):
        mem.write(b"\x00")
    with pytest.raises(LoadError, match="closed"):
        mem.protect_exec()


def test_protect_exec_prefix():
    page = mmap.PAGESIZE
    mem = ExecMemory(3 * page)
    with pytest.raises(LoadError):
        mem.protect_exec(page + 1)
    mem.protect_exec(page)
    assert mem.executable
    # The tail stays writable.
    ctypes.memset(mem.address + page, 0xAB, 16)
    assert ctypes.string_at(mem.address + page, 2) == b"\xab\xab"
    mem.close()


def test_exec_memory_errors_are_load_errors():
    with pytest.raises(LoadError, match="negative"):
        ExecMemory(-1)
    with pytest.raises(LoadError, match="cannot map"):
        ExecMemory(1 << 70)


def test_exec_memory_writes_tail_after_protect():
    page = mmap.PAGESIZE
    with ExecMemory(2 * page) as mem:
        mem.protect_exec(page)
        mem.write(b"\x01\x02", page)
        assert ctypes.string_at(mem.address + page, 2) == b"\x01\x02"
        for off in (0, page - 1):
            with pytest.raises(LoadError, match="executable"):
                mem.write(b"\x00\x00", off)


@x86_64_host
def test_x64_return_42():
    a = Assembler(x64)
    with a:
        x64.mov(x64.eax, 42)
        x64.ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_int)() == 42


@x86_64_host
def test_x64_add_two_ints():
    # First two integer arguments: rcx, rdx on Windows, rdi, rsi in SysV.
    first, second = (x64.ecx, x64.edx) if platform.system() == "Windows" else (x64.edi, x64.esi)
    a = Assembler(x64)
    with a:
        x64.mov(x64.eax, first)
        x64.add(x64.eax, second)
        x64.ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


@aarch64_host
def test_aarch64_add_two_ints():
    a = Assembler(aarch64)
    with a:
        aarch64.add(aarch64.w0, aarch64.w0, aarch64.w1)
        aarch64.ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


@riscv64_host
def test_riscv64_add_two_ints():
    a = Assembler(riscv64)
    with a:
        riscv64.addw(riscv64.a0, riscv64.a0, riscv64.a1)
        riscv64.ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


@loongarch64_host
def test_loongarch64_add_two_ints():
    a = Assembler(loongarch64)
    with a:
        loongarch64.add.w(loongarch64.a0, loongarch64.a0, loongarch64.a1)
        loongarch64.ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)(40, 2) == 42


# A METH_O builtin returning its argument: f(self, arg) tail calls
# Py_NewRef(arg), so the second argument register moves to the first.
NEWREF = Extern(ctypes.pythonapi.Py_NewRef)


def check_identity_builtin(a):
    fn = a.builtin(METH_O, name="ident")
    o = object()
    assert fn(o) is o and fn.__self__.assembler is a


@x86_64_host
def test_x64_builtin():
    first, second = (x64.rcx, x64.rdx) if platform.system() == "Windows" else (x64.rdi, x64.rsi)
    a = Assembler(x64)
    with a:
        x64.mov(first, second)
        x64.mov(x64.rax, NEWREF)
        x64.jmp(x64.rax)
    check_identity_builtin(a)


@aarch64_host
def test_aarch64_builtin():
    a = Assembler(aarch64)
    with a:
        aarch64.mov(aarch64.x0, aarch64.x1)
        aarch64.ldr(aarch64.x16, "ptr")
        aarch64.br(aarch64.x16)
        a.align(8)
        aarch64.label("ptr")
        a.qword(NEWREF)
    check_identity_builtin(a)


@riscv64_host
def test_riscv64_builtin():
    a = Assembler(riscv64)
    with a:
        riscv64.mv(riscv64.a0, riscv64.a1)
        riscv64.ld(riscv64.t0, a.extern_slot(NEWREF))
        riscv64.jr(riscv64.t0)
    check_identity_builtin(a)


@loongarch64_host
def test_loongarch64_builtin():
    a = Assembler(loongarch64)
    with a:
        loongarch64.move(loongarch64.a0, loongarch64.a1)
        loongarch64.ld.d(loongarch64.t0, a.extern_slot(NEWREF))
        loongarch64.jirl(loongarch64.zero, loongarch64.t0, 0)
    check_identity_builtin(a)
