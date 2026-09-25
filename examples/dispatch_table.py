"""A tiny bytecode interpreter with a computed-goto dispatch table.

Each opcode handler is bound to a PC label `a.pc[op]` (DynASM `=>op`). The
data section holds a table of handler addresses, written with `a.qword`
using the labels themselves, so the linker fills in absolute addresses.

Every handler ends with its own copy of the dispatch sequence (a macro), the
same "threaded" layout LuaJIT's interpreter uses.

x64 has no `jmp qword[rip + table + rcx*8]`: rip-relative addressing cannot
take an index register. So the table address is loaded once with
`lea r8, [rip + table]` and dispatch uses `jmp qword[r8 + rcx*8]`.

Bytecode format: pairs of bytes (opcode, signed 8 bit argument).
The accumulator lives in rax and is returned by HALT.

The function is generated with the `function` decorator. The body uses
the assembler it receives for `a.pc` and `a.section`, next to the module
level mnemonics.

Run with `uv run python examples/dispatch_table.py`.
"""

import ctypes

from jita import Label, function
from jita.x64 import *  # noqa: F403

# Opcodes. The value is the index into the dispatch table and the PC label.
HALT, SET, ADD, SUB, MUL = range(5)
NAMES = ["HALT", "SET", "ADD", "SUB", "MUL"]


def dispatch():
    """Macro: fetch the next (opcode, arg) pair and jump to its handler.

    rdi: bytecode pointer, r8: dispatch table, rcx: opcode, rdx: argument.
    """
    movzx(ecx, byte[rdi])
    movsx(rdx, byte[rdi + 1])
    add(rdi, 2)
    jmp(qword[r8 + rcx * 8])


@function(ctypes.c_int64, ctypes.c_char_p)
def run(a):
    # int64_t run(const uint8_t *bytecode /* rdi */)
    table = Label("dispatch_table")
    lea(r8, ptr[rip + table])  # ptr: size comes from r8
    xor(eax, eax)
    dispatch()

    label(a.pc[HALT])
    ret()

    label(a.pc[SET])
    mov(rax, rdx)
    dispatch()

    label(a.pc[ADD])
    add(rax, rdx)
    dispatch()

    label(a.pc[SUB])
    sub(rax, rdx)
    dispatch()

    label(a.pc[MUL])
    imul(rax, rdx)
    dispatch()

    # The table: one absolute 64 bit handler address per opcode.
    with a.section("data"):
        a.align(8)
        label(table)
        a.qword(*(a.pc[op] for op in range(len(NAMES))))


def main() -> None:
    program = [(SET, 3), (ADD, 4), (MUL, 5), (SUB, 1), (MUL, -2), (HALT, 0)]
    code = bytes(b for op, arg in program for b in (op, arg & 0xFF))
    result = run(code)
    assert result == ((3 + 4) * 5 - 1) * -2
    print(" ".join(f"{NAMES[op]} {arg}" for op, arg in program))
    print(f"result = {result}")


if __name__ == "__main__":
    main()
