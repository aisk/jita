"""Generate riscv64 code on any host.

Builds `int64_t sum(const int64_t *p, size_t n)` and a small function that
calls `strlen` through the extern's pointer slot, then prints the listing.
Encoding needs no riscv64 hardware; on a riscv64 host (or under
qemu-riscv64) the code is also loaded once with `a.load()` and both entry
points are called.

Run with `uv run python examples/riscv64_hello.py`.
"""

import ctypes
import platform

import jita.riscv64 as rv
from jita import Assembler, Extern
from jita.riscv64 import *  # noqa: F403  registers, mem, mnemonics
from jita.tools.listing import listing


def build() -> Assembler:
    a = Assembler(rv)  # explicit arch: works on x86 hosts too
    with a:
        # int64_t sum(const int64_t *p /* a0 */, size_t n /* a1 */)
        label("sum")
        mv(a2, zero)
        beqz(a1, "done")
        label("loop")
        ld(a3, mem[a0])  # a3 = *p
        add(a2, a2, a3)
        addi(a0, a0, 8)  # p++
        addi(a1, a1, -1)
        bnez(a1, "loop")
        label("done")
        mv(a0, a2)
        ret()

        # size_t twice_strlen(const char *s): 2 * strlen(s)
        label("twice_strlen")
        addi(sp, sp, -16)
        sd(ra, mem[sp + 8])
        ld(t0, a.extern_slot(Extern("strlen")))  # auipc + ld, any distance
        jalr(t0)
        slli(a0, a0, 1)
        ld(ra, mem[sp + 8])
        addi(sp, sp, 16)
        ret()
    return a


def main() -> None:
    a = build()
    print(listing(a))
    if platform.machine().lower() != "riscv64":
        print(f"host is {platform.machine()}, not running the code")
        return
    libc = ctypes.CDLL(None)
    strlen = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    assert strlen is not None
    mod = a.load(externs={"strlen": strlen})
    total = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, entry="sum")
    values = list(range(1, 101))
    print(f"sum(1..100) = {total((ctypes.c_int64 * 100)(*values), 100)}")
    twice = mod.function(ctypes.c_size_t, ctypes.c_char_p, entry="twice_strlen")
    print(f"twice_strlen = {twice(b'hello')}")


if __name__ == "__main__":
    main()
