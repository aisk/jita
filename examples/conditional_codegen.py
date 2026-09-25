"""Specialize code at runtime: the jita spelling of DynASM `.if`.

DynASM decides at preprocessing time with `.if`/`.endif`, and fills in
runtime values through encode-once templates. In jita the generator is
ordinary Python that runs when the program does, so a factory function
with a `function`-decorated body is all it takes. `make_reduce` returns a
reduction over an int64 array, and the body closes over the factory's
parameters:

- `op`: "add" or "max", picked by a Python `if`, not tested at runtime,
- `clamp`: when set, baked into the code as an immediate and applied
  with a cmov,
- `unroll`: 1 or 2 elements per loop iteration, emitted by a Python loop.

Every call to `make_reduce` generates and loads a new function. The short
loop bodies use `jnz.short`, which emits a rel8 branch and fails loudly at
link time if the target is ever out of range. jita never shrinks branches
on its own.

The listing of each variant is printed from `fn.assembler` and the image
of the loaded module, `fn.module.image`, so it shows real addresses.

Run with `uv run python examples/conditional_codegen.py`.
"""

import ctypes

from jita import Label, function
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


def make_reduce(op: str, clamp: int | None = None, unroll: int = 1):
    assert unroll in (1, 2)

    @function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
    def reduce(a):
        # int64_t reduce(const int64_t *p /* rdi */, size_t n /* rsi */), n % unroll == 0
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

    return reduce


def main() -> None:
    values = [7, -3, 12, 5, 9, 1]
    arr = (ctypes.c_int64 * len(values))(*values)
    variants = [
        dict(op="add"),
        dict(op="add", clamp=20, unroll=2),
        dict(op="max", unroll=2),
    ]
    results = []
    for params in variants:
        fn = make_reduce(**params)
        result = fn(arr, len(values))
        results.append(result)
        print(f"reduce({', '.join(f'{k}={v!r}' for k, v in params.items())}) = {result}")
        print(listing(fn.assembler, fn.module.image))
        print()

    assert results == [31, 20, 12]


if __name__ == "__main__":
    main()
