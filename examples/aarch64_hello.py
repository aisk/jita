"""Generate aarch64 code on any host.

Builds `int64_t sum(const int64_t *p, size_t n)` and a small function that
calls `strlen` through a pointer stored next to the code, then prints the
listing. Encoding needs no aarch64 hardware; on an aarch64 host the code
is also loaded once with `a.load()` and both entry points are called.

Run with `uv run python examples/aarch64_hello.py`.
"""

import ctypes
import platform

import jita.aarch64 as arm
from jita import Assembler, Extern
from jita.aarch64 import *  # noqa: F403  registers, mem, mnemonics
from jita.tools.listing import listing


def build() -> Assembler:
    a = Assembler(arm)  # explicit arch: works on x86 hosts too
    with a:
        # int64_t sum(const int64_t *p /* x0 */, size_t n /* x1 */)
        label("sum")
        mov(x2, xzr)
        cbz(x1, "done")
        label("loop")
        ldr(x3, mem.post[x0, 8])  # x3 = *p++
        add(x2, x2, x3)
        subs(x1, x1, 1)
        b.ne("loop")
        label("done")
        mov(x0, x2)
        ret()

        # size_t twice_strlen(const char *s): 2 * strlen(s)
        label("twice_strlen")
        stp(fp, lr, mem.pre[sp - 16])
        mov(fp, sp)
        ldr(x16, "strlen_ptr")  # pc-relative literal load
        blr(x16)
        lsl(x0, x0, 1)
        ldp(fp, lr, mem.post[sp, 16])
        ret()

        a.align(8)
        label("strlen_ptr")
        a.qword(Extern("strlen"))
    return a


def main() -> None:
    a = build()
    print(listing(a))
    if platform.machine().lower() not in ("aarch64", "arm64"):
        print(f"host is {platform.machine()}, not running the code")
        return
    libc = ctypes.CDLL(None)
    strlen = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    mod = a.load(externs={"strlen": strlen})
    total = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, entry="sum")
    values = list(range(1, 101))
    print(f"sum(1..100) = {total((ctypes.c_int64 * 100)(*values), 100)}")
    twice = mod.function(ctypes.c_size_t, ctypes.c_char_p, entry="twice_strlen")
    print(f"twice_strlen = {twice(b'hello')}")


if __name__ == "__main__":
    main()
