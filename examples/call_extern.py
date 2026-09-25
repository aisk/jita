"""Call libc's strlen from generated code, in three ways.

1. `mov rax, Extern("strlen")` loads the 64 bit address into a register
   (a movabs with an ABS64 patch), then `call rax`.
2. `call qword[rip + STRLEN]` calls through a pointer slot that jita
   creates for the extern in the read-only `externs` section. This works
   whatever the distance between the code and libc.
3. The same thing by hand: a slot in a data section holds the address
   (`a.qword(extern)`, an ABS64 patch), and the code calls through it with
   `call qword[rip + slot]`.

A plain `call(STRLEN)` would be a rel32 call, which fails to link when
libc is more than 2GB away from the code.

The three variants are entry points of one assembler, so the code is
loaded once with `a.load()` and each entry becomes a callable with
`mod.function(..., entry=name)`. `a.function` would load a separate copy
per call. Addresses of externs are supplied when loading, through the
`externs` mapping. `Extern("name", address)` fixes the address up front
instead.

Run with `uv run python examples/call_extern.py`.
"""

import ctypes

from jita import Assembler, Extern, Label
from jita.x64 import *  # noqa: F403

STRLEN = Extern("strlen")


def build() -> Assembler:
    a = Assembler()
    slot = Label()
    with a:
        # size_t via_register(const char *s)
        a.label("via_register")
        # On entry rsp is 8 mod 16; the SysV ABI wants 16 byte alignment at
        # the call, so adjust the stack around it.
        sub(rsp, 8)
        mov(rax, STRLEN)
        call(rax)
        add(rsp, 8)
        ret()

        # size_t via_extern_slot(const char *s)
        a.label("via_extern_slot")
        sub(rsp, 8)
        call(qword[rip + STRLEN])
        add(rsp, 8)
        ret()

        # size_t via_slot(const char *s)
        a.label("via_slot")
        sub(rsp, 8)
        call(qword[rip + slot])
        add(rsp, 8)
        ret()

        with a.section("data"):
            a.align(8)
            label(slot)
            a.qword(STRLEN)
    return a


def main() -> None:
    libc = ctypes.CDLL(None)
    strlen_addr = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    assert strlen_addr is not None
    # The callables keep the module alive; no `with` or close() is needed.
    mod = build().load(externs={"strlen": strlen_addr})
    sig = (ctypes.c_size_t, ctypes.c_char_p)
    via_register = mod.function(*sig, entry="via_register")
    via_extern_slot = mod.function(*sig, entry="via_extern_slot")
    via_slot = mod.function(*sig, entry="via_slot")
    s = b"hello from jita"
    n1, n2, n3 = via_register(s), via_extern_slot(s), via_slot(s)
    assert n1 == n2 == n3 == len(s)
    print(f"strlen via register: {n1}")
    print(f"strlen via extern slot: {n2}")
    print(f"strlen via slot: {n3}")


if __name__ == "__main__":
    main()
