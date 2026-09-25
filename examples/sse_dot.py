"""SSE2 dot product of two double arrays.

The main loop handles two doubles per iteration with packed instructions
(`movupd`, `mulpd`, `addpd`); a scalar tail loop (`movsd`, `mulsd`,
`addsd`) handles an odd element at the end.

Run with `uv run python examples/sse_dot.py`.
"""

import ctypes

from jita import Assembler, Label
from jita.x64 import *  # noqa: F403


def build() -> Assembler:
    # double dot(const double *x /* rdi */, const double *y /* rsi */, size_t n /* rdx */)
    a = Assembler()
    with a:
        pairs, tail, tail_loop, done = Label(), Label(), Label(), Label()
        xorpd(xmm0, xmm0)  # packed accumulator {0, 0}
        mov(rcx, rdx)
        shr(rcx, 1)  # rcx = n / 2 pairs
        jz(tail)

        label(pairs)
        movupd(xmm1, oword[rdi])  # unaligned loads: no alignment demands
        movupd(xmm2, oword[rsi])
        mulpd(xmm1, xmm2)
        addpd(xmm0, xmm1)
        add(rdi, 16)
        add(rsi, 16)
        dec(rcx)
        jnz(pairs)

        # Horizontal add: xmm0.lo += xmm0.hi
        label(tail)
        movapd(xmm1, xmm0)
        unpckhpd(xmm1, xmm1)
        addsd(xmm0, xmm1)

        and_(edx, 1)  # n % 2 elements left (0 or 1)
        jz(done)
        label(tail_loop)
        movsd(xmm1, qword[rdi])
        mulsd(xmm1, qword[rsi])
        addsd(xmm0, xmm1)
        add(rdi, 8)
        add(rsi, 8)
        dec(edx)
        jnz(tail_loop)

        label(done)
        ret()  # result in xmm0
    return a


def main() -> None:
    xs = [0.5 * i for i in range(11)]
    ys = [1.0 + i for i in range(11)]
    arr = ctypes.c_double * len(xs)
    with build().load() as mod:
        dot = mod.function(
            ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        )
        result = dot(arr(*xs), arr(*ys), len(xs))
        empty = dot(arr(), arr(), 0)
    expected = sum(x * y for x, y in zip(xs, ys))
    assert result == expected and empty == 0.0
    print(f"dot = {result}")


if __name__ == "__main__":
    main()
