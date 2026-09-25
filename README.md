# jita

jita is a Python library for writing x64 machine code by hand and running it
at runtime, at the abstraction level of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). You write instructions, labels and
data directives as ordinary Python calls; jita encodes them, links labels
and external symbols, and loads the result into executable memory that you
can call through `ctypes`. The instruction templates are a port of DynASM's
x86 table (`dasm_x86.lua`, MIT license, Copyright (C) Mike Pall), so
encodings match what DynASM would emit, apart from the differences listed
below. jita has no runtime dependencies.

## Install and run

jita needs Python 3.14 and is developed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest
uv run python examples/sum_array.py
```

The `examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, calls into libc, an SSE2 dot
product and code specialized by Python-level parameters.

## Example

```python
import ctypes
from jita import Assembler, Label
from jita.x64 import *

def build():
    a = Assembler()
    with a:                       # module level mnemonics emit into `a`
        loop, done = Label(), Label()
        xor(eax, eax)             # int64_t sum(int64_t *p, size_t n)
        test(rsi, rsi)
        jz(done)
        loop.here()
        add(rax, qword[rdi])
        add(rdi, 8)
        dec(rsi)
        jnz(loop)
        done.here()
        ret()
    return a

with build().load() as mod:
    fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
    print(fn((ctypes.c_int64 * 3)(1, 2, 3), 3))   # 6
```

## Cheat sheet

| What | jita | DynASM |
| --- | --- | --- |
| registers | `rax`, `r8d`, `al`, `xmm3`, `ymm0`, `st1`, `gp64(n)` | `rax`, `Rq(n)` |
| memory | `qword[rbx + rcx*8 + 8]` | `qword [rbx+rcx*8+8]` |
| size from the other operand | `ptr[rbx]`, `mov(rax, ptr[rbx])` | `[rbx]` |
| rip-relative | `qword[rip + lbl]`, `lea(rax, ptr[rip + lbl])` | `[->lbl]` |
| label | `lbl = Label()`, `lbl.here()`, `a.label("name")` | `1:`, `->name:` |
| label by name | `jz("done")`, `qword[rip + "tbl"]`, `a.qword("tbl")`, then `a.label("done")` | `->done`, `->tbl` |
| indexed (pc) labels | `a.pc[i]`, `a.pc[i].here()` | `=>i` |
| external symbol | `Extern("strlen")`, address given to `load(externs=...)` | `extern strlen` |
| sections | `with a.section("data"): ...`, `a.section("vars", writable=True)` | `.section` |
| data | `a.byte() a.word() a.dword() a.qword(1, lbl, ext) a.bytes(b"..") a.align(16) a.space(n)` | `.byte .dword .qword .align` |
| short branch | `jmp.short(lbl)`, `jz.short(lbl)` | automatic |
| keyword mnemonics | `and_ or_ not_ int_` | `and or not int` |
| methods instead of the context | `a.mov(rax, 1)`, `a.jmp.short(lbl)` | |

Macros are plain Python functions that emit instructions, and `.if` is a
Python `if` in the generator. Wherever a Label is accepted a string names
a label of the assembler, so `jz("done")` can come before `a.label("done")`
and the same name always means the same label. Use `Label()` objects for
labels inside macros, so that calling the macro twice does not clash.
Named labels become symbols of the loaded module
(`mod.function(..., entry="name")`, `mod.address("name")`), and a Label or
Extern used as a `mov r64` immediate or a `qword` value becomes a 64 bit
absolute address. `jita.tools.listing.listing(a)` prints the
generated code one instruction per line with its bytes and relocations.

Loaded code is read-execute, and so are sections by default. Data you want
to modify at runtime goes into a section created with `writable=True`,
which is placed on its own pages and stays read-write. Python can update
it through the module, e.g. `mod.write("counter", (5).to_bytes(8, "little"))`.
`a.align(n)` pads code with NOPs and data sections with zero bytes.

`call(Extern(...))` is a rel32 call, so loading fails with `LinkError` when
the target is more than 2GB away from the code. In that case load the
address into a register first: `mov(rax, Extern("f")); call(rax)`.

Scalar SSE instructions need an explicitly sized memory operand, e.g.
`mulsd(xmm0, qword[rip + k])`; `ptr[...]` does not pick the size there.

## Differences from DynASM

- No silent branch relaxation. `jmp`/`jcc` to a label always use rel32.
  `jmp.short` always uses rel8 and linking fails if the target is too far.
- Immediates are range checked by value. `add rax, 0xffffffff` is an
  error instead of quietly becoming `add rax, -1`. `mov r64, imm` switches
  to the 64 bit `movabs` form when the value does not fit in 32 bits.
- A few fixes where DynASM's x64 output is wrong: `xchg eax, eax` is
  `87 C0` rather than the `90` nop, and 32 bit address registers (`[eax]`)
  get the `0x67` prefix.
- Some instructions DynASM lacks are added: `xadd`, `cmpxchg`,
  `cmpxchg8b`, `cmpxchg16b`, `ud2`, `hlt` and the `movsq`/`cmpsq`/`stosq`/
  `lodsq`/`scasq` string ops. They combine with `lock()` and `rep()`.
- Everything is known when an instruction is encoded, so there is no
  preprocessor, no action list and no separate link step to call by hand.

## Status

x64 only. Tested on Linux. macOS uses the same mmap/mprotect path but is
untested, and executable memory on Windows is not implemented yet. aarch64
is planned next.
