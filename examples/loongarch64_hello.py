"""Generate loongarch64 code on any host.

Builds `int64_t sum(const int64_t *p, size_t n)` and a small function that
calls `strlen` through the extern's pointer slot, then prints the listing.
Encoding needs no loongarch64 hardware; on a loongarch64 host (or under
qemu-loongarch64) the code is also loaded once with `a.load()` and both
entry points are called.

Run with `uv run python examples/loongarch64_hello.py`.
"""

import ctypes
import platform

import jita.loongarch64 as la64
from jita import Assembler, Extern
from jita.loongarch64 import *  # noqa: F403  registers and mnemonics
from jita.tools.listing import listing


def build() -> Assembler:
    a = Assembler(la64)  # explicit arch: works on x86 hosts too
    with a:
        # int64_t sum(const int64_t *p /* a0 */, size_t n /* a1 */)
        label("sum")
        move(a2, zero)
        beqz(a1, "done")
        label("loop")
        ld.d(a3, a0, 0)  # a3 = *p
        add.d(a2, a2, a3)
        addi.d(a0, a0, 8)  # p++
        addi.d(a1, a1, -1)
        bnez(a1, "loop")
        label("done")
        move(a0, a2)
        ret()

        # size_t twice_strlen(const char *s): 2 * strlen(s)
        label("twice_strlen")
        addi.d(sp, sp, -16)
        st.d(ra, sp, 8)
        ld.d(t0, a.extern_slot(Extern("strlen")))  # pcalau12i + ld.d, any distance
        jirl(ra, t0, 0)
        slli.d(a0, a0, 1)
        ld.d(ra, sp, 8)
        addi.d(sp, sp, 16)
        ret()
    return a


def main() -> None:
    a = build()
    print(listing(a))
    if platform.machine().lower() != "loongarch64":
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
