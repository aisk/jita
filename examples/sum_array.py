"""Sum an array of int64 values.

Shows the basic workflow in jita:

- the `function` decorator, which runs the decorated body once inside a
  fresh Assembler and turns the name into a ctypes callable,
- module level mnemonic functions (`mov`, `add`, ...) emitting into that
  assembler,
- labels defined with `label("name")` and referenced by name,
- a macro, which is just a Python function that emits instructions.

Run with `uv run python examples/sum_array.py`.
"""

import ctypes

from jita import function
from jita.x64 import *  # noqa: F403  registers, size prefixes, mnemonics


def load_next(dst, ptr):
    """Macro: dst = *ptr++ (one qword). DynASM would spell this `|.macro`."""
    mov(dst, qword[ptr])
    add(ptr, 8)


@function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
def sum_array(a):
    # int64_t sum_array(const int64_t *p /* rdi */, size_t n /* rsi */)
    xor(eax, eax)  # acc = 0 (writing eax clears the upper half of rax)
    test(rsi, rsi)
    jz("done")  # n == 0: nothing to do, forward reference by name

    label("loop")
    load_next(rcx, rdi)
    add(rax, rcx)
    dec(rsi)
    jnz("loop")  # backward jump, rel32 unless you ask for .short

    label("done")
    ret()


def main() -> None:
    values = list(range(1, 101))
    arr = (ctypes.c_int64 * len(values))(*values)
    total = sum_array(arr, len(values))
    assert total == sum(values)
    print(f"sum(1..100) = {total}")


if __name__ == "__main__":
    main()
