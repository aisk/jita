"""One generator, different machine code: the jita spelling of DynASM `.if`.

DynASM decides at preprocessing time with `.if`/`.endif`. In jita the
generator is ordinary Python, so plain `if` statements, loops and function
parameters select what gets emitted. Here `gen_reduce` builds a reduction
over an int64 array, specialized by:

- `op`: "add" or "max", picked by the generator, not tested at runtime,
- `clamp`: when set, the result is clamped to `clamp` with a cmov,
- `unroll`: 1 or 2 elements per loop iteration.

The short loop bodies use `jnz.short`, which emits a rel8 branch and fails
loudly at link time if the target is ever out of range. jita never shrinks
branches on its own.

The listing tool shows the code each variant produced.

Run with `uv run python examples/conditional_codegen.py`.
"""

import ctypes

from jita import Assembler, Label
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403

INT64_MIN = -(1 << 63)


def combine(op, acc, val):
    """Macro: acc = op(acc, val)."""
    if op == "add":
        add(acc, val)
    elif op == "max":
        cmp(acc, val)
        cmovl(acc, val)
    else:
        raise ValueError(f"unknown op {op!r}")


def gen_reduce(a: Assembler, op: str, clamp: int | None = None, unroll: int = 1) -> None:
    # int64_t reduce(const int64_t *p /* rdi */, size_t n /* rsi */), n % unroll == 0
    assert unroll in (1, 2)
    with a:
        loop, done = Label(), Label()
        if op == "add":
            xor(eax, eax)
        else:
            mov(rax, INT64_MIN)  # does not fit in int32, so this is a movabs
        test(rsi, rsi)
        jz.short(done)
        label(loop)
        for i in range(unroll):
            combine(op, rax, qword[rdi + 8 * i])
        add(rdi, 8 * unroll)
        sub(rsi, unroll)
        jnz.short(loop)
        label(done)
        if clamp is not None:
            mov(rcx, clamp)
            cmp(rax, rcx)
            cmovg(rax, rcx)
        ret()


def run(values: list[int], **params) -> tuple[int, str]:
    a = Assembler()
    gen_reduce(a, **params)
    arr = (ctypes.c_int64 * len(values))(*values)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
        # With the linked image the listing shows real addresses and branch bytes.
        return fn(arr, len(values)), listing(a, mod.image)


def main() -> None:
    values = [7, -3, 12, 5, 9, 1]
    variants = [
        dict(op="add"),
        dict(op="add", clamp=20, unroll=2),
        dict(op="max", unroll=2),
    ]
    for params in variants:
        result, text = run(values, **params)
        print(f"reduce({', '.join(f'{k}={v!r}' for k, v in params.items())}) = {result}")
        print(text)
        print()

    assert run(values, op="add")[0] == 31
    assert run(values, op="add", clamp=20, unroll=2)[0] == 20
    assert run(values, op="max", unroll=2)[0] == 12


if __name__ == "__main__":
    main()
