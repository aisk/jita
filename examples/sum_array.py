"""Sum an array of int64 values.

Shows the basic DynASM workflow in jita:

- an Assembler used as a context manager, so the module level mnemonic
  functions (`mov`, `add`, ...) emit into it,
- labels defined with `label("name")` and referenced by name,
- a macro, which is just a Python function that emits instructions,
- loading the code and calling it through ctypes.

Run with `uv run python examples/sum_array.py`.
"""

import ctypes

from jita import Assembler
from jita.x64 import *  # noqa: F403  registers, size prefixes, mnemonics


def load_next(dst, ptr):
    """Macro: dst = *ptr++ (one qword). DynASM would spell this `|.macro`."""
    mov(dst, qword[ptr])
    add(ptr, 8)


def build() -> Assembler:
    # int64_t sum(const int64_t *p /* rdi */, size_t n /* rsi */)
    a = Assembler()
    with a:
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
    return a


def main() -> None:
    values = list(range(1, 101))
    arr = (ctypes.c_int64 * len(values))(*values)
    with build().load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
        total = fn(arr, len(values))
    assert total == sum(values)
    print(f"sum(1..100) = {total}")


if __name__ == "__main__":
    main()
