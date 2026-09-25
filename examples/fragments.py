"""Encode once, instantiate many times: fragments with holes.

A Fragment is encoded like any other code, except that some operands are
`Hole`s: registers, immediates and labels whose values are supplied later.
`frag.instantiate(...)` copies the encoded bytes and patches the holes in,
without running the encoder again, which is what DynASM's `dasm_put` does
with `Rq(n)` registers and runtime immediates.

Here `STEP` is one Horner step, `acc = acc * x + c`, and `SUM` sums an int64
array with a loop whose label is local to each instance. The function built
below evaluates a polynomial with several STEP instances and adds the sums
of two arrays, each computed by an instance of SUM using other registers.

Run with `uv run python examples/fragments.py`.
"""

import ctypes

from jita import Assembler, Fragment, Hole, Label
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403

acc, x, c = Hole.gp64("acc"), Hole.gp64("x"), Hole.imm32("c")
STEP = Fragment()
with STEP:
    imul(acc, x)
    add(acc, c)

total, src, n, done = Hole.gp64("total"), Hole.gp64("src"), Hole.gp64("n"), Hole.label("done")
SUM = Fragment()
with SUM:
    xor(total, total)
    test(n, n)
    jz(done)
    label("loop")  # a fresh label in every instance
    add(total, qword[src + n * 8 - 8])
    dec(n)
    jnz.short("loop")


def build(coeffs: list[int]) -> Assembler:
    """int64_t f(int64_t x, int64_t *a, int64_t na, int64_t *b, int64_t nb)
    returns poly(x) + sum(a) + sum(b), with poly's coefficients baked in."""
    a = Assembler()
    with a:
        mov(rax, coeffs[0])
        for coef in coeffs[1:]:
            STEP.instantiate(acc=rax, x=rdi, c=coef)
        after_a, after_b = Label(), Label()
        SUM.instantiate(total=r9, src=rsi, n=rdx, done=after_a)
        label(after_a)
        add(rax, r9)
        SUM.instantiate(total=r10, src=rcx, n=r8, done=after_b)
        label(after_b)
        add(rax, r10)
        ret()
    return a


def main() -> None:
    coeffs = [3, -2, 5, 7]  # 3x^3 - 2x^2 + 5x + 7
    asm = build(coeffs)
    print(f"STEP is {len(STEP)} bytes, SUM is {len(SUM)} bytes, holes {list(SUM.holes)}")
    print(listing(asm))
    xs = (ctypes.c_int64 * 3)(1, 2, 3)
    ys = (ctypes.c_int64 * 2)(100, 200)
    I64 = ctypes.c_int64
    with asm.load() as mod:
        f = mod.function(I64, I64, ctypes.c_void_p, I64, ctypes.c_void_p, I64)
        result = f(10, ctypes.addressof(xs), 3, ctypes.addressof(ys), 2)
    print(f"poly(10) + 6 + 300 = {result}")


if __name__ == "__main__":
    main()
