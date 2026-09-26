"""`ExecMemory` on every host, plus calling convention free smoke tests of
loaded code on x86-64 and aarch64 hosts, Windows included. The rest of the
runtime tests write SysV code and live in test_runtime.py."""

import ctypes
import mmap
import platform

import pytest

import jita.aarch64 as aarch64
import jita.x64 as x64
from jita import Assembler, LoadError
from jita.runtime import ExecMemory

MACHINE = platform.machine().lower()
x86_64_host = pytest.mark.skipif(MACHINE not in ("x86_64", "amd64"), reason="needs an x86-64 host")
aarch64_host = pytest.mark.skipif(MACHINE not in ("aarch64", "arm64"), reason="needs an aarch64 host")


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
